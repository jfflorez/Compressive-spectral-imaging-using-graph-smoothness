import sys, os
try:
    thisFilePath = os.path.abspath(__file__)
except NameError:
    print("Error: __file__ is not available. 'thisFilePath' will resolved to os.getcwd().")
    thisFilePath = os.getcwd()  # Use current directory or specify a default

projectPath = os.path.normpath(os.path.join(thisFilePath, "..",'..'))  # Move up to project root

if projectPath not in sys.path:  # Avoid duplicate entries
    sys.path.append(projectPath)

import yaml, json
import re
import tempfile
from typing import Union
from copy import deepcopy
import h5py

from sensing_models.dual_cam_sd_cassi import DualCameraSDCassiModel
from reconst_algs.gcsi_bbreconst_core import  ingestion_process, worker_task
from reconst_algs.gcsi_bbreconst_core import  generate_block_reconst_tasks
import numpy as np

import multiprocessing as mp
#from multiprocessing import Manager, Pool, Process
from functools import partial
import psutil
PHYSICAL_CORES  = psutil.cpu_count(logical=False) # maybe we can reduce it by 20 percent!

import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def prepare_tmp_dir(path_to_file):
    tmp_dir = os.path.splitext(path_to_file)[0]
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def create_init_msg(model_obj, blocks, tmp_dir):
    """
    Create the init message dict for ingestion and save it atomically
    in the tmp_dir as 'init_msg.json' for standalone recovery.
    """
    dataset_name = model_obj.get_dataset_name()
    n1, n2, L = model_obj.n1, model_obj.n2, model_obj.L

    init_msg = {
        'title': f'Reconstruction of dataset: {dataset_name}',
        'shape': (n1, n2, L),
        'dtype': 'float64',
        'dataset_name': dataset_name,
        'number_of_blocks': len(blocks)
    }

    # Ensure tmp_dir exists
    os.makedirs(tmp_dir, exist_ok=True)
    init_msg_path = os.path.join(tmp_dir, 'init_msg.json')

    # Atomic write
    with tempfile.NamedTemporaryFile('w', delete=False, dir=tmp_dir, encoding='utf-8') as tmp_file:
        json.dump(init_msg, tmp_file, indent=2)
        tmp_temp_path = tmp_file.name

    # Rename to final path (atomic on most OS)
    os.replace(tmp_temp_path, init_msg_path)

    return init_msg


def run_worker_pool(blocks, model_obj, tmp_dir, queue, n_procs, graph_params, solver_params):

    worker_fn = partial(worker_task,
                        output_dir=tmp_dir,
                        DCSDCassiModelObj=model_obj,
                        graph_type=graph_params['graph_type'],
                        graph_params=graph_params,
                        solver_params = solver_params,
                        queue_obj=queue)
    if n_procs > 1:
        with mp.Pool(processes=n_procs) as pool:
            pool.map(worker_fn, blocks)
    else:
        for block in blocks:
            worker_fn(block)

def next_versioned_path(
    dataset_name,
    results_dir="results",
    suffix="_reconst_",
    ext_override=".h5"
    ):
    """
    Create a non-overwriting, versioned filepath:
    <base><suffix>v<version><ext>

    dataset_name (str) : is the file path of the cassi dataset to be processed
    """
    
    # Get dataset's filename. This discards everything (the base path) before the last slash
    filename = os.path.split(dataset_name)[1]
    base, ext = os.path.splitext(filename)
    ext = ext_override or ext

    pattern = re.compile(
        rf"^{re.escape(base)}{re.escape(suffix)}v(\d+){re.escape(ext)}$"
        # ^   start of filename
        # ()  capture group for version number
        # \d+ one or more digits
        # $   end of filename
    )

    max_version = -1

    for fname in os.listdir(results_dir):
        m = pattern.match(fname)
        if m:
            max_version = max(max_version, int(m.group(1)))

    next_version = max_version + 1

    return os.path.join(
        results_dir,
        f"{base}{suffix}v{next_version}{ext}"
    )


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------
def main(config_dict_or_path: Union[dict, str] = None):


    """
    Main entry point for experiments.
    `config_dict_or_path` can be:
        - dict: full configuration
        - str: path to a YAML config file
    """

    # 1. Default parameters
    defaults = {
        'results_dir': 'results/',
        'dataset_name': 'datasets/simulated_data_HSDC2_DB_Oct112019_2_OE.mat',
        'number_of_processors': 4,
        'block_width': 32,
        'block_height': 32,
        'block_overlap': 0.5,
        'graph_params': {'graph_type': 'ROPs'},
        'display_slice': 5,
        'solver_params': {'alpha': 7.19/2, 'maxiter': 10000, 'tol': 1e-7, 'noisy_meas': False}
    }

    # 2. Handle config input
    config = deepcopy(defaults)  # always start with defaults

    if config_dict_or_path is not None and isinstance(config_dict_or_path, str):
        # config is a YAML file path
        if not os.path.isfile(config_dict_or_path):
            raise FileNotFoundError(f"Config file not found: {config_dict_or_path}")
        with open(config_dict_or_path, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        # Merge YAML on top of defaults
        config.update(yaml_cfg)
    elif config_dict_or_path is not None and isinstance(config, dict):
        # config is a dict → merge on top of defaults
        config.update(config_dict_or_path)
    else:
        raise TypeError("config must be a dict or a path to a YAML file")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    
    path_to_dataset = os.path.normpath(config['dataset_name']).replace(os.sep, '/')

    if not os.path.isfile(path_to_dataset):
        raise FileNotFoundError(f'{path_to_dataset} could not be found. Place dataset file in datasets/ folder')

    path_to_file = next_versioned_path(path_to_dataset,
                                       config['results_dir'],
                                       suffix="_reconst_",
                                       ext_override=".h5")
    tmp_dir = prepare_tmp_dir(path_to_file)

    # ------------------------------------------------------------------
    # Instantiate model object
    # ------------------------------------------------------------------
    dcsdcassi_model_obj = DualCameraSDCassiModel(path_to_dataset)
    dcsdcassi_model_obj.prepare_for_pickle()  # remove large attributes for multiprocessing

    # ------------------------------------------------------------------
    # Generate blocks
    # ------------------------------------------------------------------
    blocks = list(generate_block_reconst_tasks(
        dcsdcassi_model_obj.sdcassi_obj,
        config['block_width'],
        config['block_height'],
        config['block_overlap']
    ))

    # ------------------------------------------------------------------
    # Initialize queue and ingestion process and run worker pool
    # ------------------------------------------------------------------
    with mp.Manager() as manager:
        queue = manager.Queue()
        # TODO: Maybe send the input config parameters via init_msg.update(config) too?
        init_msg = create_init_msg(dcsdcassi_model_obj, blocks, tmp_dir)
        init_msg.update(config)
        queue.put(init_msg)

        # ---------------------------------------------------------------
        # Launch ingestion process
        # ---------------------------------------------------------------
        ing_obj = mp.Process(target=ingestion_process, args=(path_to_file, queue),
                             name="data-ingestion")
        ing_obj.start()
        

        # ------------------------------------------------------------------
        # Worker pool
        # ------------------------------------------------------------------

        # Cap the requested number of processors if greater than actual number of physical cores
        n_procs = min(config['number_of_processors'], PHYSICAL_CORES)

        try:
            run_worker_pool(blocks, dcsdcassi_model_obj, tmp_dir, queue, n_procs,
                            config['graph_params'], config['solver_params'],
                            )
            # when n_procs = 1, the main excecution path takes care of the tasks (sequential processing of tasks)
        finally:
            # Ensure ingestion process receives sentinel and cleanup
            queue.put(None)
            ing_obj.join(timeout=60)
            if ing_obj.is_alive():
                print("Ingestion process timed out, terminating...")
                ing_obj.terminate()
                ing_obj.join()

    # ------------------------------------------------------------------
    # Check X_hat and if possible add reconstruction metrics to HDF5 results
    # ------------------------------------------------------------------
    if os.path.isfile(path_to_file):
        # Load ground truth spectral image
        X_gt = dcsdcassi_model_obj.load_X()

        with h5py.File(path_to_file, 'r+') as f:
            if 'X_hat' not in f:
                raise KeyError("Dataset 'X_hat' not found in HDF5 file. Data ingestion may have failed.")
            X_hat = f['X_hat'][:]
            if X_gt is not None:
                from utils.metrics import evaluate_metrics
                metrics_dict, sam_map, ssim_map = evaluate_metrics(X_gt,np.maximum(X_hat, 0))
                print("\n".join([f"{key}:{value}" for key, value in metrics_dict.items()]))
                f.attrs.update(metrics_dict) 

                f.create_dataset('iqa_images/ssim_map', data=ssim_map)  
                f.create_dataset('iqa_images/sam_map', data=sam_map)  

                plt.figure(figsize=(12,4))
                plt.subplot(1,2,1)
                plt.imshow(sam_map, cmap="inferno")
                plt.colorbar(label="SAM (degrees)")
                plt.title("Spectral Angle Mapper")

                plt.subplot(1,2,2)
                plt.imshow(ssim_map, cmap="viridis", vmin=0, vmax=1)
                plt.colorbar(label="avg SSIM")
                plt.title("Structural Similarity")

                plt.tight_layout()
                plt.show()       


        X_hat = np.maximum(X_hat, 0)

        plt.figure()
        plt.imshow(X_hat[:, :, config['display_slice']], cmap='gray')
        plt.show()

    print("End of program")

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
    #path_to_dataset = 'datasets/simulated_data_HSDC1_DB_Oct092019_5_OE.mat'
    #path_to_dataset_reconst = 'results/simulated_data_HSDC1_DB_Oct092019_5_OE_reconst_v19.h5'

    #import utils.datasets as datasetManager
    #import utils.metrics as metrics
    # Load reference spectral image 
    #X = datasetManager.load_dataset(path_to_dataset)['X']

    #with h5py.File(path_to_dataset_reconst,mode='r') as f:
    #    X_hat = f['X_hat'][:]

    #metrics_dict = metrics.evaluate_metrics(X,X_hat)

    #print(metrics_dict)



