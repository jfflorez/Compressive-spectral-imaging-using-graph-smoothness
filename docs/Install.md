## Install Python Environment Using Miniforge and conda-forge

We recommend using Miniforge to manage your conda environments. Miniforge ensures compatibility with packages from the conda-forge channel.

Make sure you have installed Miniforge.

### Open Miniforge Prompt

⚠️ Ensure your Conda base environment is from Miniforge (not Anaconda). Run conda info and check for miniforge in the base path and conda-forge as the default channel.

Create the Environment from environment.yml. Inside the Miniforge Prompt or a terminal with access to conda and run:

```
cd path/to/GitHub/Compressive-spectral-imaging-using-graph-smoothness
conda env create --file environment.yml
```

