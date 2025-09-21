import os
import sys
import fire
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from shapely.geometry import Polygon
from pyproj import Proj, Transformer
import descarteslabs as dl
from tqdm import tqdm
import concurrent.futures

sys.path.append('util')
from constants import *

def read_metadata(metadata_file):
    """Get a list of lats, lons, and loss years"""
    metadata = pd.read_csv(metadata_file)
    data_points = metadata[[LATITUDE_HEADER, LONGITUDE_HEADER, YEAR_HEADER]].to_numpy()
    return data_points

def get_scenes_from_latlon(lat, lon, resolution, pad):
    """Get tiles and scenes from lat lons"""
    tile = dl.scenes.DLTile.from_latlon(lat=lat, lon=lon,
                                        resolution=resolution,
                                        tilesize=2,
                                        pad=pad)
    scenes, ctx = dl.scenes.search(aoi=tile,
                                   products=[GLOBAL_FOREST_CHANGE_PRODUCT_ID])
    return tile, scenes, ctx

def get_threshold_im(a):
    """Format np array so it is readable by findContours"""
    a = a.astype(np.uint8)
    a = np.transpose(a, (1, 2, 0))
    retval, threshold = cv2.threshold(a, 0, 1, cv2.THRESH_BINARY)
    return retval, threshold

def get_loss_data(scene, ctx, start_year, end_year):
    """Return a dictionary of np array loss masks by year"""
    lossyear = scene.ndarray(LOSSYEAR_BAND, ctx, mask_nodata=True)
    treecover = scene.ndarray(TREECOVER_BAND, ctx, mask_nodata=True)
    enough_trees = np.where(treecover.data > FORESTED_THRESHOLD, 1, 0)
    loss_by_year = lossyear.data * enough_trees
    losses = {}
    for year in range(start_year, end_year + 1):
        losses[year] = np.where(loss_by_year == (year - BASE_YEAR), 1, 0)
    return losses

def get_shapes_for_coordinates(lat, lon, resolution=30, pad=0,
                               start_year=2012, end_year=2018):
    """
    Retrieves a list of (shape, lat, lon, year) tuples for every
    shape in the given DLTile.
    """
    tile, scenes, ctx = get_scenes_from_latlon(lat, lon, resolution, pad)
    losses = get_loss_data(scenes[0], ctx, start_year, end_year)
    all_shape_info = []
    for year in range(start_year, end_year + 1):
        ret, thresh = get_threshold_im(losses[year])
        outlines, contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
        for outline in outlines:
            # skip if not enough pixels in shape
            if len(outline) <= SHAPE_SIZE_THRESHOLD:
                continue
            outline = np.squeeze(outline)
            shape = Polygon(outline)
            shape_datum = {'shape': shape, 'year': year}
            all_shape_info.append(shape_datum)

    return all_shape_info

def download_polygons(lat, lon, loss_year, resolution, pad, polygon_dir):
    """Downloads all polygon data between 2012 and 2018 (inclusive) for a lat-lon pair"""
    loss_year = int(loss_year)
    shape_data = get_shapes_for_coordinates(lat, lon, resolution, pad,
                                            start_year=BASE_YEAR,
                                            end_year=END_YEAR)
    shape_data = sorted(shape_data, key=lambda shape_datum: shape_datum['year'])
    idx_counter = 0
    seen_years = set([])

    for shape_datum in shape_data:
        shape = shape_datum['shape']
        year = shape_datum['year']

        if year not in seen_years:
            seen_years.add(year)
            idx_counter = 0

        shapefile_path = polygon_dir / f'{round(lat, 5)}_{round(lon, 5)}'
        shapefile_path = shapefile_path / f'{year}'
        if not os.path.isdir(shapefile_path):
            os.makedirs(shapefile_path)
        shapefile_path = shapefile_path / f'shape_{idx_counter}'
        pickle.dump(shape, open(shapefile_path, 'wb'))
        idx_counter += 1
    
    return len(shape_data)

def download_polygons_parallel(split, polygon_dir, max_workers=8):
    """Downloads all polygon data between 2012 and 2018 (inclusive) using multithreading"""
    tile_size_km = HANSEN_TILE_SIZE_KM
    resolution = LANDSAT8_TIER1_PRODUCT_RES
    tile_size = int(tile_size_km * 1000 / resolution)
    pad = int(tile_size / 2) - 1

    polygon_dir = Path(polygon_dir)
    metadata_file = METADATA_FILES[split]
    data_points = read_metadata(metadata_file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        downloaded_polygons = \
            {executor.submit(download_polygons, 
                             lat, 
                             lon, 
                             loss_year, 
                             resolution, 
                             pad, 
                             polygon_dir): 
                             (lat, lon, loss_year) for lat, lon, loss_year in data_points}
        
        for future in tqdm(concurrent.futures.as_completed(downloaded_polygons), total=len(data_points)):
            num = future.result()

if __name__ == "__main__":
    fire.Fire(download_polygons_parallel)
