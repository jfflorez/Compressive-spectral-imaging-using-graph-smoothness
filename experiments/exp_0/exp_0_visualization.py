import os, sys
from pathlib import Path

project_path = str(Path(__file__).parent.parent)
if project_path not in sys.path:
    sys.path.append(project_path)

# Data processing and scientific computing
import re
import h5py
from h5py import AttributeManager

import yaml
import pandas as pd
import numpy as np


# Visualization
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import matplotlib.pyplot as plt

# Experiment management 
from utils.myems.myems.experiments import Experiment, flatten_dict

params_dir = Path('experiments/runtime_eval_bb_reconstructoin/parameters')
results_dir = Path('experiments/runtime_eval_bb_reconstructoin/outputs')

def get_attrs_dict(group : h5py.Group):
    
    attrs_dict ={}
    attrs_dict['group_name'] = group.name
    for key, value in group.attrs.items():
        # convert bytes → str if needed
        if isinstance(value, bytes):
            value = value.decode("utf-8")   

        attrs_dict[f"{key}"] = value
    return attrs_dict 


def load_metadata(exp_name :str = "exp_0", files_to_analyze = None) :
    # Analysis script
    with Experiment(
        name=exp_name,  # Same name as computation experiment
        params_baseline=None,  # Signal: analysis mode
        params_to_study=None,
        base_dir = os.path.join(project_path,"experiments") 
        ) as exp:
        # Load results from existing outputs
        #results = load_results(exp.outputs_dir)    


        hash_re = re.compile(r"_([0-9a-fA-F]+)\.yaml$")
        rows = []
        # Back track output file based on associated input parameter hash
        for params_file in exp.params_dir.glob("**/*.yaml"):
            
            m = hash_re.search(params_file.name)
            if not m:
                continue
            hash_value = m.group(1)
            #print(params_file.name, hash_value)

            
            matches = list(
                exp.outputs_dir.glob(f"**/*_{hash_value}_v*.h5")
            )
            print(params_file.name, hash_value,matches)
            if not matches:
                continue  # or raise

            # Load YAML once per params file
            with open(params_file, "r") as f:
                params = yaml.safe_load(f) or {}
            
            params_flat = flatten_dict(params)

            for output_file in matches:

                if files_to_analyze is not None and output_file.name not in files_to_analyze:
                    continue

                prefix_row = {}

                # Add identifiers (very useful later)
                prefix_row["hash"] = hash_value
                prefix_row["params_file"] = params_file.name
                prefix_row["output_file"] = str(output_file.relative_to(project_path))

                # Add YAML params
                prefix_row.update(params_flat)

                #print(f"{params_file.name}, \n{output_file.name}")
                with h5py.File(output_file, "r") as f:
                    #f.name
                    root_attrs_dict = get_attrs_dict(f)
                    #prefix_row.update(attrs_dict)
                    #for key, value in f.attrs.items():
                        # convert bytes → str if needed
                    #    if isinstance(value, bytes):
                    #        value = value.decode("utf-8")                        
                    #    prefix_row[f"h5.{key}"] = value
                    row = {}
                    #row.update(prefix_row)
                    for group_name, group_obj in f["block_estimates"].items():
                        row = {}
                        row.update(prefix_row)
                        row.update(root_attrs_dict)
                        #attrs_dict = get_attrs_dict(group_obj)
                        row.update(get_attrs_dict(group_obj))
                        rows.append(row)
        #        print('\n')
        df = pd.DataFrame(rows)



        return df

def generate_figures(df):
    import numpy as np
    """receives a dataframe df, where rows represent block spectral images estimates and columns their attributes"""

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Add overall title
    fig.suptitle('Runtimes vs iterations for block spectral image estimates', fontsize=14, y=0.98)

    graph_types = df['graph_graph_type'].unique()
    cmap = matplotlib.colormaps.get_cmap('Accent')
    colors = cmap(np.arange(0,cmap.N,step=cmap.N//np.size(graph_types)))
    #get_cmap('Accent', len(graph_types))
    #coeff_of_variation = df['solver_info_y_var'] / df['solver_info_y_mean']
    for i, graph_type in enumerate(graph_types):
        subset = df[df['graph_graph_type'] == graph_type]
        
        ax[0].scatter(
            x=subset['solver_info_num_iters'],
            y=subset['elpased_time_signal_reconst'],
            s= 10*subset.solver_info_y_var/subset.solver_info_y_mean,
            color=colors[i],
            label=graph_type
        )

    block_sizes = sorted(df['block_estimate_size'].unique())
    cmap = matplotlib.colormaps.get_cmap('viridis_r')
    colors = cmap(np.arange(0,cmap.N,step=cmap.N//np.size(block_sizes)))
    for i, block_size in enumerate(block_sizes):
        subdf = df[df['block_estimate_size']==block_size]
        ax[1].scatter(
                x=subdf['solver_info_num_iters'],
                y=subdf['elpased_time_signal_reconst'],
                s= 10*subdf.solver_info_y_var/subdf.solver_info_y_mean,
                color=colors[i],
                label=block_size
                #c=df['block_estimate_size'],
                #cmap='viridis_r'
                )

    ax[0].set_ylabel('Reconstruction time (seconds)')
    ax[0].set_xlabel('Solver iterations \n(to convergence)')
    ax[0].legend(title='Graph type')

    ax[1].set_xlabel('Solver iterations \n(to convergence)')
    ax[1].legend(title='Block size')

    #norm = plt.Normalize(df['block_estimate_size'].min(), df['block_estimate_size'].max())
    #fig.colorbar(mtlb.cm.ScalarMappable(norm=norm, cmap='viridis_r'), ax=ax[1], label='Block Estimate Size')

    plt.tight_layout()
    plt.show()


    # Plot patch locations on the measurement domain with dot sizes and color representing
    # the patch's coeff. of variation and the associated block size, respectively 

    fig2, ax2 = plt.subplots(figsize=(12, 5),nrows=1,ncols=2)

    # Add overall title
    fig2.suptitle('Spatial locations of measurement patches', fontsize=14, y=0.98)

    coeff_of_variation = df['solver_info_y_var'] / df['solver_info_y_mean']

    ax2[0].scatter(
        x=df.block_x0,
        y=df.block_y0,
        #s=10 * df.block_estimate_size/df.block_estimate_size.max(),
        c=coeff_of_variation,
        cmap='viridis_r')    

    ax2[0].set_ylabel('y-coordinate')
    ax2[0].set_xlabel('x-coordinate')
    #ax2[0].xaxis.set_inverted(True)
    ax2[0].yaxis.set_inverted(True)

    ax2[1].scatter(
    x=df.block_x0,
    y=df.block_y0,
    #s=10 * df.block_estimate_size/df.block_estimate_size.max(),
    c=df.block_estimate_size,
    cmap='viridis_r')    

    ax2[1].set_ylabel('y-coordinate')
    ax2[1].set_xlabel('x-coordinate')
    #ax2[1].xaxis.set_inverted(True)
    ax2[1].yaxis.set_inverted(True)

    norm = plt.Normalize(coeff_of_variation.min(), coeff_of_variation.max())
    fig2.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis_r'), ax=ax2[0], label='Block Estimate Size')


    norm = plt.Normalize(df['block_estimate_size'].min(), df['block_estimate_size'].max())
    fig2.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis_r'), ax=ax2[1], label='Block Estimate Size')


    fig2.tight_layout()

    #fig2.show()    

    fig_1_dict = {'figure' : fig, 'name' : 'figure_runtime_vs_iterations' }
    fig_2_dict = {'figure' : fig2, 'name' : 'figure_patch_locations' }

    return [fig_1_dict,fig_2_dict]


import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
from functools import lru_cache

# Cache .mat loads to avoid re-reading same file multiple times
@lru_cache(maxsize=None)
def load_mat_cube(path):
    mat = loadmat(f'{project_path}/{path}')
    # Adjust key if needed
    return mat["X"]

def compute_mse(df) -> pd.DataFrame:
    df = df.copy()
    mse = []
    block_size = []
    for index, row in df.iterrows():
        # Load ground truth cube
        X = load_mat_cube(row["dataset_file"])

        # Open reconstruction file
        output_file = row["output_file"]
        with h5py.File(f'{project_path}/{output_file}', "r") as f:
            grp = f[row["group_name"]]
            
            x_hat = grp["x_hat"][()]          # reconstructed block
            multi_idx = grp["multi_idx"][()]  # indices
            

        # Ensure integer indexing
        x_coords,y_coords,z_coords = tuple(multi_idx.astype(int))

        # Extract corresponding ground truth block
        X_sub = X[x_coords,y_coords,z_coords]
        # Compute MSE
        mse.append(np.mean((X_sub - x_hat) ** 2))
        block_size.append(x_hat.size)
        
    df['mse'] = mse
    df['block_size'] = block_size
    return df


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def generate_parallel_efficiency_figure(df: list[dict]) -> plt.Figure:
    """
    Build a parallel efficiency figure from df produced by load_metadata().

    Result from evaluating compute_parallel_efficiency_metrics(df)
    Parameters 
    ----------
    results : list of dict, each with keys:
        - graph_type       : str
        - pool_size        : int
        - total_compute    : float
        - real_time        : float
        - ideal_time       : float
        - parallel_efficiency : float  (0-1)
        - overhead         : float
        - serial_fraction  : float     (0-1)

    Returns
    -------
    fig : matplotlib.Figure
    """
    results = compute_parallel_efficiency_metrics(df)

    graph_types = sorted(set(r['graph_type'] for r in results))
    pool_sizes  = sorted(set(r['pool_size']  for r in results))

    # index results for easy lookup
    data = {(r['graph_type'], r['pool_size']): r for r in results}

    n_groups  = len(graph_types)
    n_workers = len(pool_sizes)
    x         = np.arange(n_groups)
    width     = 0.35

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    colors_real  = ['#2c7bb6', '#abd9e9']
    colors_ideal = ['#d7191c', '#fdae61']

    # ── Panel 1: Real vs Ideal runtime ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for i, ps in enumerate(pool_sizes):
        real_vals  = [data[(gt, ps)]['real_time']  for gt in graph_types]
        ideal_vals = [data[(gt, ps)]['ideal_time'] for gt in graph_types]
        offset = (i - (n_workers - 1) / 2) * width
        ax1.bar(x + offset - width/4, real_vals,  width/2, label=f'Real  Workers={ps}',  color=colors_real[i],  alpha=0.9)
        ax1.bar(x + offset + width/4, ideal_vals, width/2, label=f'Ideal Workers={ps}', color=colors_ideal[i], alpha=0.9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(graph_types,fontsize=12)
    ax1.set_ylabel('Time (s)', fontsize=12)
    ax1.set_title('Real vs Ideal Runtime', fontsize=14)
    ax1.legend(fontsize=10)

    # ── Panel 2: Parallel efficiency ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bar_colors = ['#1a9641', '#a6d96a', '#fdae61', '#d7191c']
    bars = []
    bar_labels = []
    for i, ps in enumerate(pool_sizes):
        eff_vals = [data[(gt, ps)]['parallel_efficiency'] * 100 for gt in graph_types]
        offset = (i - (n_workers - 1) / 2) * width
        b = ax2.bar(x + offset, eff_vals, width * 0.9,
                    color=bar_colors[i % len(bar_colors)], alpha=0.9)
        bars.append(b)
        bar_labels.append(f'Workers={ps}')
        for rect, val in zip(b, eff_vals):
            ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax2.axhline(100, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.set_ylim(0, 115)
    ax2.set_xticks(x)
    ax2.set_xticklabels(graph_types,fontsize=12)
    ax2.set_ylabel('Parallel Efficiency (%)',fontsize=12)
    ax2.set_title('Parallel Efficiency',fontsize=14)
    ax2.legend(bars, bar_labels, fontsize=10)

    # ── Panel 3: Overhead (absolute) ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for i, ps in enumerate(pool_sizes):
        oh_vals = [data[(gt, ps)]['overhead'] for gt in graph_types]
        offset = (i - (n_workers - 1) / 2) * width
        ax3.bar(x + offset, oh_vals, width * 0.9,
                color=bar_colors[i % len(bar_colors)], alpha=0.9, label=f'Workers={ps}')

    ax3.set_xticks(x)
    ax3.set_xticklabels(graph_types,fontsize=12)
    ax3.set_ylabel('Overhead (s)',fontsize=12)
    ax3.set_title('Real Wallclock Runtime - Ideal Runtime',fontsize=14)
    ax3.legend(fontsize=10)

    # ── Panel 4: Serial fraction ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for i, ps in enumerate(pool_sizes):
        sf_vals = [data[(gt, ps)]['serial_fraction'] * 100 for gt in graph_types]
        offset = (i - (n_workers - 1) / 2) * width
        b = ax4.bar(x + offset, sf_vals, width * 0.9,
                    color=bar_colors[i % len(bar_colors)], alpha=0.9, label=f'Workers={ps}')
        for rect, val in zip(b, sf_vals):
            ax4.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.2,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    ax4.set_xticks(x)
    ax4.set_xticklabels(graph_types,fontsize=12)
    ax4.set_ylabel('Serial Fraction (%)',fontsize=12)
    ax4.set_title('Serial / Overhead Fraction',fontsize=14)
    ax4.legend(fontsize=10)

    fig.suptitle('Parallel Producer-Consumer Architecture — Efficiency Analysis', fontsize=13, fontweight='bold')
    return {'figure' : fig, 'name' : "figure_parallel_eff_analysis"}


def compute_parallel_efficiency_metrics(df):
    
    """df generated by load_metadata(exp.name)"""

    results = []


    for group_key, sub_df in df.groupby(['dataset_file', 'graph_graph_type', 'number_of_processors']):

        
        dataset_file, graph_type, pool_size = group_key    

        sub_df = sub_df.copy()
        sub_df['dataset_file'] = dataset_file
        sub_df = compute_mse(sub_df)  # apply per group directly

        total_compute = (
            sub_df['elpased_time_graph_inference'] +
            sub_df['elpased_time_signal_reconst']
        ).sum()

        real_time = sub_df['aggregation_duration'].iloc[0]
        ideal_time = total_compute / pool_size
        parallel_efficiency = ideal_time / real_time
        overhead = real_time - ideal_time
        serial_fraction = overhead / real_time

        sam = sub_df['SAM'].iloc[0]
        ssim = sub_df['SSIM'].iloc[0]
        psnr = sub_df['PSNR'].iloc[0]

        print(f"\nGraph: {graph_type}, Workers: {pool_size}")
        print(f"Total compute: {total_compute:.2f}s")
        print(f"Real wallclock runtime: {real_time:.2f}s")
        print(f"Ideal parallel runtime: {ideal_time:.2f}s")
        print(f"Parallel efficiency: {parallel_efficiency:.2%}")
        print(f"Overhead: {overhead:.2f}s")
        print(f"Serial fraction: {serial_fraction:.2%}")
        print(f"SAM :{sam:.2}")
        print(f"PSNR :{psnr:.2}")
        print(f"SSIM :{ssim:.2}")

        row = {'graph_type' : graph_type,
            'pool_size' : pool_size,
            'total_compute' : total_compute,
            'real_time' : real_time,
            'ideal_time' : ideal_time,
            'parallel_efficiency' : parallel_efficiency,
            'overhead' : overhead,
            'serial_fraction' : serial_fraction}
        results.append(row)

    return results 