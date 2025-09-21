

# Deforestation Global
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>

## Metadata structure for vision models

- `DATA_BASE_DIR / curtis_processed / v5_<split>_all`: metadata *prior to* the image download script. This includes **loss areas**. 

-- Scripts used: 

- `DATA_BASE_DIR / curtis_v5 / postdownload_meta_<split>_v5`: metadata *immediately after* the image download script. This includes all columns from the csv above, with additional columns tracking information about downloaded image files.
 
 -- Scripts used: `data/download_images.py`

- `DATA_BASE_DIR / curtis_v5 / <split>_v5`: . This will be the metadata file fed into the training pipeline, will contain additional metadata columns such as continent-headers, etc.. Currently set to `postdownload_meta_<split>_v5`
 
 -- Scripts used: `data/intermediate_module.py`

## Useful links 
- [Negative sampling (Google Map API)](https://github.com/stanfordmlgroup/old-starter/blob/master/preprocess/get_negatives.py)
- [Example of dataset implementation: USGS dataset](https://github.com/stanfordmlgroup/old-starter/blob/master/data/usgs_dataset.py)
- [Documentation for Fire](https://github.com/google/python-fire/blob/master/docs/guide.md)
- [Documentation for Pytorch Lighning](https://williamfalcon.github.io/pytorch-lightning/)
