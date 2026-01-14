## Spectral image reconstrution for compressive spectral imaging using smoothness on graphs

This repository contains both real and simulated datasets to test the methods and reproduce the results of our paper:

1. **Block-based spectral image reconstruction using smoothness on graphs.**

Python implementations of reconstruction methods and usage examples are still being refined so please check constantly for updates.

## Project requirements:
* [Intructions to set up a suitable Python environment](docs/Install.md)

## Get started:

Reconstruct a spectral image from a real dual camera SDCASSI dataset as follows:

``` 
python main_reconst_from_real_measurements.py
```

You should get the following visualization:

<img src="figures\result_visualizations_for_real_data_SCN_2_scale_2_June032021_OE.svg" alt="Architecture" width="300"/>


## Citation

The datasets are available for non-commercial research use. If you use our datasets in an academic publication, kindly cite the following paper:
```
@article{florez2022block,
  title={Block-based spectral image reconstruction for compressive spectral imaging using smoothness on graphs},
  author={Florez-Ospina, Juan F and Alrushud, Abdullah KM and Lau, Daniel L and Arce, Gonzalo R},
  journal={Optics Express},
  volume={30},
  number={5},
  pages={7187--7209},
  year={2022},
  publisher={Optica Publishing Group}
}
```
