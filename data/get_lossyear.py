import sys
import fire
import numpy as np
import pandas as pd
import scipy
from PIL import Image
import descarteslabs as dl
from tqdm import tqdm

sys.path.append('util')
from constants import *

def get_latlon_from_metadata(metadata_file):
    metadata = pd.read_csv(metadata_file)
    latlon_arr = metadata[[LATITUDE_HEADER, LONGITUDE_HEADER]].to_numpy()
    return latlon_arr, metadata

def get_scenes_from_latlon(lat, lon, resolution, pad):
    tile = dl.scenes.DLTile.from_latlon(lat=lat, lon=lon,
                                        resolution=resolution,
                                        tilesize=2,
                                        pad=pad)
    scenes, ctx = dl.scenes.search(aoi=tile,
                                   products=[GLOBAL_FOREST_CHANGE_PRODUCT_ID])
    return scenes, ctx

def get_lossyear_from_latlon(lat, lon, resolution, pad):
    scenes, ctx = get_scenes_from_latlon(lat, lon, resolution, pad)
    scene = scenes[0]
    lossyear = scene.ndarray('lossyear', ctx, mask_nodata=True)
    lossyear = np.squeeze(lossyear, axis=0)
    lossyear_data = np.ma.compressed(lossyear)
    lossyear_mode = scipy.stats.mode(lossyear_data, axis=None).mode
    return -1 if len(lossyear_mode) == 0 else lossyear_mode[0] + BASE_YEAR
    
def get_lossyear(metadata_file, output_metadata_file):
    lossyears = []
    latlon_arr, metadata = get_latlon_from_metadata(metadata_file)
    
    tile_size_km = HANSEN_TILE_SIZE_KM
    resolution = LANDSAT8_TIER1_PRODUCT_RES
    tile_size = int(tile_size_km * 1000 / resolution)
    pad = int(tile_size / 2) - 1
    
    lossyears = [get_lossyear_from_latlon(lat, lon, resolution, pad) for lat, lon in tqdm(latlon_arr)]
    metadata[YEAR_HEADER] = lossyears
    metadata.to_csv(output_metadata_file, index=False)

if __name__ == "__main__":
    fire.Fire(get_lossyear)
