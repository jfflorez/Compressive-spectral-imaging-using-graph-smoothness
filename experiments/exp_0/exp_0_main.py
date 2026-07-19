"""
Block-Based Reconstruction Runtime Evaluation
==============================================

Comparison of block based reconstruction runtimes as we change 
number of processors and the graph inference method.
"""

import sys
import os
from pathlib import Path
import multiprocessing as mp

# ============================================================================
# PATH SETUP
# ============================================================================
project_path = str(Path(__file__).parent.parent)
if project_path not in sys.path:
    sys.path.append(project_path)

# ============================================================================
# IMPORTS - BEFORE EXPERIMENT CONTEXT
# ============================================================================
# CRITICAL: Import your pipeline BEFORE entering the Experiment context
# This ensures multiprocessing setup happens cleanly
from reconst_algs.gcsi_bbreconst_alg import main as my_analysis_pipeline
from utils.myems.myems.experiments import Experiment


# ============================================================================
# CONFIGURATION
# ============================================================================
PARAMS_BASELINE_PATH = os.path.normpath(
    os.path.join(project_path, 'reconst_algs/gcsi_bbreconst_config.yaml')
)

PARAMETERS_TO_STUDY = {
    'number_of_processors': [1,4],
    'graph_params': [
        {'graph_type': 'ROPs'}, 
        {'graph_type': 'Kalofolias', 'num_neigs': 33}
    ]
}


# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================
def run_experiment():
    """Run the block-based reconstruction experiment"""
    
    with Experiment(
        name="runtime_eval_bb_reconstruction",
        params_baseline=PARAMS_BASELINE_PATH,
        params_to_study=PARAMETERS_TO_STUDY,
        description=(
            'Comparison of block based reconstruction runtimes as we '
            'change number of processors and the graph inference method.'
        )
    ) as exp:
        
        print(f"\n{exp.summary()}\n")
        print(f"Running {exp.metadata['num_runs']} configurations...\n")
        
        # Iterate over parameter grid
        
        for i, params_variant in enumerate(exp.parameter_grid):

            # Unpack params_variant into params_hash : str, params : dict
            params_hash, params =  params_variant

            skip_run = any([(params_hash in file) for file in os.listdir(exp.outputs_dir)])
            if skip_run:
                print(f"Parameters with hash {params_hash} were already processed.")
                print(f"Reprocess them by deleting the corresponding YAML file and rerunning script.")
                continue
                        
            print(f"Run {i+1}/{exp.metadata['num_runs']}: ", end="")
            print(f"number_of_processors={params['number_of_processors']}, "
                  f"graph_params={params['graph_params']}")
            
            # Run the pipeline, and save results to output dir : exp.outputs_dir
            try:
                # Update parameter suffix with 
                params['suffix'] = params_hash
                results = my_analysis_pipeline(params, exp.outputs_dir)
                
                # Save results if pipeline returns something
                if results is not None:
                    exp.save_output(results, name=f"run_{i:03d}")
                
                print(f"  ✓ Completed successfully")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                # Save error info
                exp.save_output(
                    {"error": str(e), "params": params},
                    name=f"run_{i:03d}_FAILED"
                )
        
        print(f"\n{'='*80}")
        print(f"Experiment completed!")
        print(f"Results saved to: {exp.experiment_dir}")
        print(f"{'='*80}\n")


# ============================================================================
# MAIN ENTRY POINT - CRITICAL
# ============================================================================
if __name__ == '__main__':
    # Required for Windows/macOS with multiprocessing
    #mp.freeze_support()
    
    # If you're still having issues, try setting start method explicitly:
    # mp.set_start_method('spawn', force=True)
    
    # Run the experiment
    run_experiment()

