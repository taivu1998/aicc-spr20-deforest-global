import os
import sys
import fire
import numpy as np
import pandas as pd
import pickle
import glob
from tqdm import tqdm

sys.path.append('util')
from constants import *

def read_metadata(metadata_file):
    metadata = pd.read_csv(metadata_file)
    metadata_arr = metadata[[LATITUDE_HEADER, LONGITUDE_HEADER, YEAR_HEADER]].to_numpy()
    return metadata_arr, metadata

def get_loss_areas(lat, lon, loss_year, polygon_dir):
    loss_areas = []
    shapefile_path = polygon_dir / f'{round(lat, 5)}_{round(lon, 5)}'

    for year in range(BASE_YEAR, END_YEAR + 1):
        loss_area = 0
        shapefile_path_year = shapefile_path / f'{year}'
        if os.path.isdir(shapefile_path_year):
            for shape_file in glob.glob(str(shapefile_path_year) + '/*'):
                with open(shape_file, 'rb') as f:
                    polygon = pickle.load(f)
                    loss_area += polygon.area
        #loss_area /= MAX_LOSS_AREA
        loss_areas.append(loss_area)

    return loss_areas

def save_polygon_loss_areas(split, output_metadata_file):
    metadata_file = METADATA_FILES[split]
    polygon_dir = POLYGON_DIRS[split]
    metadata_arr, metadata = read_metadata(metadata_file)
    loss_area_data = [get_loss_areas(lat, lon, loss_year, polygon_dir)
                      for lat, lon, loss_year in tqdm(metadata_arr)]
    for year in range(BASE_YEAR, END_YEAR + 1):
        loss_areas_in_year = [loss_areas[year - BASE_YEAR] for loss_areas in loss_area_data]
        metadata[f'loss_area_{year}'] = loss_areas_in_year
    metadata.to_csv(output_metadata_file, index=False)

if __name__ == "__main__":
    fire.Fire(save_polygon_loss_areas)
