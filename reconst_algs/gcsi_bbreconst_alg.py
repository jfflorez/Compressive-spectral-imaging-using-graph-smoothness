import sys, os
try:
    thisFilePath = os.path.abspath(__file__)
except NameError:
    print("Error: __file__ is not available. 'thisFilePath' will resolved to os.getcwd().")
    thisFilePath = os.getcwd()  # Use current directory or specify a default

projectPath = os.path.normpath(os.path.join(thisFilePath, "..",'..'))  # Move up to project root

if projectPath not in sys.path:  # Avoid duplicate entries
    sys.path.append(projectPath)

import h5py
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Manager
from functools import partial
import psutil
import yaml, json
import re

from sensing_models.dual_cam_sd_cassi import DualCameraSDCassiModel
from reconst_algs.gcsi_bbreconst_core import  ingestion_process, worker_task
from reconst_algs.gcsi_bbreconst_core import  generate_block_reconst_tasks

import numpy as np
# and dcsd_cassi.DualCameraSDCassiModel are imported or defined elsewhere

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def prepare_tmp_dir(path_to_file):
    tmp_dir = os.path.splitext(path_to_file)[0]
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir

import json
import os
import tempfile

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


def launch_ingestion(path_to_file, queue_obj):
    ing_obj = mp.Process(target=ingestion_process, args=(path_to_file, queue_obj),
                         name="data-ingestion")
    ing_obj.start()
    return ing_obj

def run_worker_pool(blocks, model_obj, tmp_dir, queue, n_procs, graph_type, graph_params, solver_params):
    worker_fn = partial(worker_task,
                        output_dir=tmp_dir,
                        DCSDCassiModelObj=model_obj,
                        graph_type=graph_type,
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
    """

    base, ext = os.path.splitext(dataset_name)
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
def main():
    # ------------------------------------------------------------------
    # Configuration dictionary (parameters centralized)
    # ------------------------------------------------------------------
    config = {
        'config_file': None,                  # Optional YAML file path
        'dataset_dir': 'datasets/',
        'results_dir': 'results/',
        'dataset_name': 'simulated_data_HSDC1_DB_Oct092019_5_OE.mat', #'real_data_SCN_1_scale_2_June032021_OE.mat',
        'number_of_processors': 4,
        'block_width': 32,
        'block_height': 16,
        'graph_type': 'Kalofolias', #'ROPs',#'ROPs, Kalofolias',
        'graph_params': {},                    # e.g., {'num_neigs': 33}
        'display_slice': 25,                   # slice for plotting
        'solver_params' : {'alpha': 7.19, 'maxiter': 10000, 'tol': 1e-6, 'noisy_meas': False}
    }

    # Optional: load config from YAML
    if config['config_file'] and os.path.isfile(config['config_file']):
        with open(config['config_file'], 'r') as f:
            cfg = yaml.safe_load(f)
            config.update(cfg)  # override defaults

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    path_to_dataset = os.path.normpath(os.path.join(config['dataset_dir'], config['dataset_name'])).replace(os.sep, '/')
    path_to_file = next_versioned_path(config['dataset_name'],
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
        config['block_height']
    ))

    # ------------------------------------------------------------------
    # Initialize queue and ingestion process
    # ------------------------------------------------------------------
    with Manager() as manager:
        queue = manager.Queue()
        init_msg = create_init_msg(dcsdcassi_model_obj, blocks, tmp_dir)
        queue.put(init_msg)

        ing_obj = launch_ingestion(path_to_file, queue)

        # Determine number of processors
        try:
            #physical_cores = mp.cpu_count()
            physical_cores  = psutil.cpu_count(logical=False) # maybe we can reduce it by 20 percent!
        except NotImplementedError:
            physical_cores = config['number_of_processors']
        n_procs = min(config['number_of_processors'], physical_cores)

        # ------------------------------------------------------------------
        # Worker pool
        # ------------------------------------------------------------------
        try:
            run_worker_pool(blocks, dcsdcassi_model_obj, tmp_dir, queue, n_procs,
                            config['graph_type'], config['graph_params'], config['solver_params'],
                            )
            # when n_procs = 1, the main excecution path takes care of the tasks (sequential processing of tasks)
        finally:
            # Ensure ingestion process receives sentinel and cleanup
            queue.put(None)
            ing_obj.join(timeout=10)
            if ing_obj.is_alive():
                print("Ingestion process timed out, terminating...")
                ing_obj.terminate()
                ing_obj.join()

    # ------------------------------------------------------------------
    # Post-process HDF5 results
    # ------------------------------------------------------------------
    if os.path.isfile(path_to_file):
        with h5py.File(path_to_file, 'r') as f:
            if 'X_hat' not in f:
                raise KeyError("Dataset 'X_hat' not found in HDF5 file. Data ingestion may have failed.")
            X_hat = f['X_hat'][:]

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
