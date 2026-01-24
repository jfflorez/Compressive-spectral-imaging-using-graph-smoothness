import sys, os
try:
    thisFilePath = os.path.abspath(__file__)
except NameError:
    print("Error: __file__ is not available. 'thisFilePath' will resolved to os.getcwd().")
    thisFilePath = os.getcwd()  # Use current directory or specify a default

projectPath = os.path.normpath(os.path.join(thisFilePath, "..",'..'))  # Move up to project root

if projectPath not in sys.path:  # Avoid duplicate entries
    sys.path.append(projectPath)

import numpy as np
import matplotlib.pyplot as plt

from sensing_models.sd_cassi import SingleDisperserCassiModel, SignalSubDomain
from sensing_models.dual_cam_sd_cassi import DualCameraSDCassiModel
from reconst_algs.gcsi_algs import gsm_noiseless_case_estimation, gsm_noisy_case_estimation
from graphs.structure_learning import build_kalofolias_graph_adj_mtrx, build_knn_graph_adj_mtrx, build_rop_graph_adj_mtrx

import time
import h5py
import json
import re


from typing import Optional
import multiprocessing as mp

def run_block_reconst_algo(
    DCSDCassiModelObj: DualCameraSDCassiModel, 
    block: tuple[int,int,int,int], 
    graph_type: str,
    graph_params: dict = None,
    solver_params: dict = None
):
    # Set defaults
    graph_params = graph_params or {}
    solver_params = solver_params or {'tol': 1e-6, 'maxiter': 20000, 'noisy_meas' : True}
    
    # Unpack measurement patch parameters
    x0, y0, height, width = block
    n1, n2, L = (getattr(DCSDCassiModelObj, name) for name in ['n1','n2','L'])
    
    # Define CASSI system of linear equations for a given block
    t_start = time.perf_counter()
    H_k, y_k, omega_k = DCSDCassiModelObj.get_linear_system(block)
    #Omega_tilde_k, Omega_k = DCSDCassiModelObj.sdcassi_obj.get_system_submtx_pair(k=(x0,y0,height,width))
    t_end = time.perf_counter()
    print(f'Submatrix extraction elapsed time: {t_end - t_start}')
    
    side_img = DCSDCassiModelObj.Z
    
    # Build graph with method-specific params
    t_start_graph_inf = time.perf_counter()
    match graph_type:
        case 'Kalofolias':
            W_G = build_kalofolias_graph_adj_mtrx(
                side_img, omega_k, 
                spectral_img_shape=(n1, n2, L),
                **graph_params
            ).tocsr()
        case 'ROPs':
            W_G = build_rop_graph_adj_mtrx(side_img, omega_k, **graph_params).tocsr()
        case 'knn':
            W_G = build_knn_graph_adj_mtrx(
                side_img, omega_k,
                spectral_img_shape=(n1, n2, L),
                **graph_params
            ).tocsr()
        case _:
            raise NotImplementedError(
                f"graph_type must be in ['Kalofolias', 'ROPs', 'knn']. Got: {graph_type}."
            )
    t_end_graph_inf = time.perf_counter()
    
    # Run spectral image estimation 
    t_start_signal_reconst = time.perf_counter()
    if 'noisy_meas' in solver_params and solver_params['noisy_meas']==True:
        # TODO: make alpha a user defined hyperparameter
        x_hat, solver_info = gsm_noisy_case_estimation(H_k, y_k, W_G, params=solver_params)
    else: # assume noiseless measurements and perfect system matrix
        x_hat, solver_info = gsm_noiseless_case_estimation(H_k, y_k, W_G, params=solver_params)
    t_end_signal_reconst = time.perf_counter()

    multi_idx = omega_k.unravel_index(order='F')
    #np.unravel_index(omega_k.to_array(), (n1, n2, L) , order='F')

    

    metadata = {'elpased_time_graph_inference' : t_end_graph_inf- t_start_graph_inf,
                'elpased_time_signal_reconst' : t_end_signal_reconst- t_start_signal_reconst
               }
      
    metadata.update({
        'block_id' : int(np.ravel_multi_index((y0, x0), DCSDCassiModelObj.Y.shape, order='F')),
        'block_x0' : x0, 'block_y0': y0, 'block_height': height, 'block_width': width,
        'graph_type': graph_type,
    })
    

    # Flatten nested dicts with prefixes
    for key, val in graph_params.items():
        metadata[f'graph_{key}'] = val
    for key, val in solver_params.items():
        metadata[f'solver_{key}'] = val
    for key, val in solver_info.items():
        metadata[f'solver_info_{key}'] = val
    
    return x_hat, multi_idx, metadata

def save_block_reconst_results(output_filepath: str,
                               data: tuple[np.ndarray, np.ndarray, dict]):
    """ save output from run_block_reconst_algo() """

    # Verify extension and fix it if incorrect
    if not output_filepath.endswith('.h5'):
        output_filepath = os.path.splitext(output_filepath)[0] + '.h5'

    # Atomic save : either we finish writing the h5 file and rename the *.tmp.h5 or it stays
    # as temporary and it hints something wrong occur during the writing process. 
    tmp_path = output_filepath + '.tmp'

    x_hat, multi_idx, metadata = data
    with h5py.File(tmp_path, mode='w') as f:
        f.create_dataset('x_hat', data=x_hat)
        f.create_dataset('multi_idx', data=multi_idx)
        f.attrs.update(metadata)

    print(f"File written, renaming...")  # DEBUG
    os.replace(tmp_path, output_filepath)
    print(f"File saved to: {output_filepath}")  # DEBUG

    return output_filepath

def generate_block_reconst_tasks(
    SDCassiModelObj: SingleDisperserCassiModel, 
    width: int, 
    height: int, 
    overlap: float = 0
) -> list[tuple[int, int, int, int]]:
    """
    Generate block tasks for block-based spectral image reconstruction.
    
    Creates a grid of overlapping or non-overlapping rectangular patches that cover
    the snapshot image dimensions, accounting for dispersion in the x-direction.
    
    Parameters
    ----------
    SDCassiModelObj : SingleDisperserCassiModel
        CASSI model object containing spectral image configuration.
    width : int
        Width of each reconstruction block (x-direction).
    height : int
        Height of each reconstruction block (y-direction).
    overlap : float, optional
        Fractional overlap between adjacent blocks, by default 0.
        Must be in range [0, 1). For example, 0.5 means 50% overlap.
    
    Returns
    -------
    list[tuple[int, int, int, int]]
        List of tuples (x0, y0, width, height) defining each block's position
        and size. x0, y0 are the top-left corner coordinates.
    
    Notes
    -----
    The x-dimension (n2) is extended by (L-1) to account for spectral dispersion.
    Blocks are generated in row-major order (iterating over x, then y).
    """
    n1, n2, L = SDCassiModelObj.get_spectral_image_shape()
    m2 = n2+L-1
    xx = np.arange(0, m2, step = (1 - overlap) * width, dtype=int)
    yy = np.arange(0, n1, step = (1 - overlap) * height, dtype=int)

    return [(x0, y0, min(height,n1-y0), min(width,m2-x0)) for x0 in xx for y0 in yy]  

# ---------------------------------------------------------------------------
# Block-based spectral image reconstruction worker   
# ---------------------------------------------------------------------------

def worker_task(block, output_dir, DCSDCassiModelObj, graph_type, graph_params, solver_params, queue_obj):
    """
    Worker task for reconstructing a block and sending metadata to the ingestion queue.
    
    Notes:
    - Results themselves are not sent via queue to avoid large object serialization.
    - Each worker saves its block results to a separate HDF5 file atomically.
    """

    # ------------------------------------------------------------------
    # Validate output directory
    # ------------------------------------------------------------------
    dataset_name = os.path.splitext(getattr(DCSDCassiModelObj, 'dataset_name'))[0]
    if dataset_name not in output_dir:
        raise ValueError(f"Output directory '{output_dir}' does not match dataset '{dataset_name}'.")

    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Output directory does not exist: {output_dir}")

    # ------------------------------------------------------------------
    # Block info
    # ------------------------------------------------------------------
    x0, y0, height, width = block
    block_id = int(np.ravel_multi_index((y0, x0), DCSDCassiModelObj.Y.shape, order='F'))
    output_filepath = os.path.join(output_dir, f"block_{block_id}.h5")
    output_filepath = os.path.normpath(output_filepath)

    # ------------------------------------------------------------------
    # Prepare metadata
    # ------------------------------------------------------------------
    metadata = {
        'block_id': block_id,
        'block_x0': x0,
        'block_y0': y0,
        'block_height': height,
        'block_width': width,
        'graph_type': graph_type,
    }

    # ------------------------------------------------------------------
    # Skip if block already exists
    # ------------------------------------------------------------------
    if os.path.isfile(output_filepath):
        print(f"Task for block {block_id} already completed. File: {output_filepath}")
        metadata.update({
            'elapsed_time': 0.0,
            'elapsed_time_units': 'seconds',
            'output_filepath': output_filepath
        })
        queue_obj.put(metadata)
        return

    # ------------------------------------------------------------------
    # Run reconstruction algorithm
    # ------------------------------------------------------------------
    start_time = time.perf_counter()

    # Example solver params — could be passed via config
    #solver_params = {'alpha': 7.19, 'maxiter': 10000, 'tol': 1e-4, 'noisy_meas': True}

    x_hat, multi_idx, algo_metadata = run_block_reconst_algo(
        DCSDCassiModelObj, block, graph_type, graph_params, solver_params
    )

    # ------------------------------------------------------------------
    # Save block results (atomic if save_block_reconst_results does it)
    # ------------------------------------------------------------------
    output_filepath = save_block_reconst_results(output_filepath, (x_hat, multi_idx, algo_metadata))

    # ------------------------------------------------------------------
    # Update metadata and send to queue
    # ------------------------------------------------------------------
    end_time = time.perf_counter()
    metadata.update({
        'elapsed_time': end_time - start_time,
        'elapsed_time_units': 'seconds',
        'output_filepath': output_filepath
    })

    queue_obj.put(metadata)


# ------------------------------------------------------------------
# Core ingestion worker: queue mandatory, no auto-resume
# ------------------------------------------------------------------
def ingestion_process(path_to_file: str, q: mp.Queue):
    """
    Child process to ingest and aggregate block data into a single HDF5 file.
    Queue must be provided by main process.

    Contract:
    - First message in queue must be init_msg dict containing:
        'title', 'shape', 'dtype', 'number_of_blocks', 'dataset_name'
    - Subsequent messages: {'output_filepath': str, 'block_id': str|int}
    - Queue signals end of ingestion by sending None
    """
    if q is None:
        raise ValueError("Queue must be provided for ingestion_process")

    print('----- Start of Data Ingestion Process --------')

    # Receive init message
    init_msg = q.get()
    if init_msg is None:
        print("No init message received. Aborting ingestion.")
        return

    # Start timing
    start_time = time.perf_counter()
    print("Start timing")

    # Temporary HDF5 file
    path_root = os.path.splitext(path_to_file)[0]
    path_to_tmpfile = path_root + ".tmp.h5"

    # Ensure tmp HDF5 exists
    if not os.path.isfile(path_to_tmpfile):
        with h5py.File(path_to_tmpfile, "w") as aggr_file:
            aggr_file.attrs["title"] = init_msg["title"]
            aggr_file.attrs["shape"] = list(init_msg["shape"])
            aggr_file.attrs["dtype"] = str(init_msg["dtype"])
            aggr_file.attrs["number_of_blocks"] = init_msg["number_of_blocks"]

            n1, n2, L = aggr_file.attrs["shape"]
            dtype = np.dtype(init_msg["dtype"])
    else:
        with h5py.File(path_to_tmpfile, "r") as aggr_file:
            n1, n2, L = tuple(aggr_file.attrs["shape"])
            dtype = np.dtype(aggr_file.attrs["dtype"])

    # ------------------------------------------------------------------
    # Aggregation loop
    # ------------------------------------------------------------------
    with h5py.File(path_to_tmpfile, "r+") as aggr_file:
        X_hat = np.zeros((n1, n2, L), dtype=np.float64)
        C = np.zeros((n1, n2, L), dtype=np.int32)

        total_num_blocks = aggr_file.attrs["number_of_blocks"]
        block_counter = 0

        while True:
            result = q.get()  # blocking
            if result is None:
                break

            output_filepath = result["output_filepath"]
            block_id = result["block_id"]
            group_name = f"block_{block_id}"

            with h5py.File(output_filepath, "r") as f:
                x_hat = f["x_hat"][:]
                multi_idx = tuple(f["multi_idx"][:])

                # Copy block only once (idempotent)
                if group_name not in aggr_file:
                    aggr_file.copy(source=f, dest=group_name)
                    block_counter += 1

            X_hat[multi_idx] += x_hat
            C[multi_idx] += 1

        # Safe division
        X_hat = np.divide(X_hat, C, out=np.zeros_like(X_hat), where=C > 0)

        if "X_hat" in aggr_file:
            del aggr_file["X_hat"]
        aggr_file.create_dataset("X_hat", data=X_hat)

        end_time = time.perf_counter()
        aggr_file.attrs["aggregation_time"] = (
            aggr_file.attrs.get("aggregation_time", 0.0) + (end_time - start_time)
        )

    # Promote tmp file if complete
    if block_counter == total_num_blocks:
        os.replace(path_to_tmpfile, path_to_file)

    print('----- End of Data Ingestion Process --------')
    print(f"Data ingestion elapsed time: {end_time - start_time}")

# ------------------------------------------------------------------
# Standalone wrapper: iterates tmp dir, queue-free
# ------------------------------------------------------------------
def standalone_ingestion_process(path_to_file: str):
    """
    Standalone ingestion: aggregates partial blocks in tmp dir.
    Safe for manual recovery or debugging. No queue used.
    """
    print('----- Start of Standalone Ingestion --------')

    path_root = os.path.splitext(path_to_file)[0]
    tmp_dir = path_root
    path_to_tmpfile = path_root + ".tmp.h5"

    # Check init message
    init_msg_path = os.path.join(tmp_dir, "init_msg.json")
    if not os.path.isfile(init_msg_path):
        print(f"Missing init message: {init_msg_path}")
        return

    with open(init_msg_path, "r") as f:
        init_msg = json.load(f)

    n1, n2, L = init_msg["shape"]
    dtype = np.dtype(init_msg["dtype"])
    total_num_blocks = init_msg["number_of_blocks"]

    # Open tmp HDF5
    with h5py.File(path_to_tmpfile, "w") as aggr_file:
        aggr_file.attrs.update(init_msg)

        X_hat = np.zeros((n1, n2, L), dtype=np.float64)
        C = np.zeros((n1, n2, L), dtype=np.int32)

        # Find all block files
        pattern = re.compile(r"^block_(\d+)\.h5$")
        block_counter = 0

        for fname in os.listdir(tmp_dir):
            match = pattern.match(fname)
            if not match:
                continue

            block_id = match.group(1)
            group_name = f"block_{block_id}"
            filepath = os.path.join(tmp_dir, fname)

            with h5py.File(filepath, "r") as f:
                x_hat = f["x_hat"][:]
                multi_idx = tuple(f["multi_idx"][:])
                if group_name not in aggr_file:
                    aggr_file.copy(source=f, dest=group_name)
                    block_counter += 1
                X_hat[multi_idx] += x_hat
                C[multi_idx] += 1

        # Safe division
        X_hat = np.divide(X_hat, C, out=np.zeros_like(X_hat), where=C > 0)

        if "X_hat" in aggr_file:
            del aggr_file["X_hat"]
        aggr_file.create_dataset("X_hat", data=X_hat)

    # Promote tmp file if complete
    if block_counter == total_num_blocks:
        os.replace(path_to_tmpfile, path_to_file)

    print('----- End of Standalone Ingestion --------')
    print(f"Aggregated {block_counter}/{total_num_blocks} blocks")

