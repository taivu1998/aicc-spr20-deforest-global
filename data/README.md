# To download images from the DescartesLabs api before training

### Downloads training, validation and test splits:
`python download_images.py --split train`

`python download_images.py --split val`

`python download_images.py --split test`


All images saved to `DATA_BASE_DIR / curtis_v5 / <split>` in `util/constants.py`. Metadata saved to same folder.


### To process downloaded metadata before training: 
`python intermediate_module.py`

Processed metadata saved to `DATA_BASE_DIR / curtis_v5`
