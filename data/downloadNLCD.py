import descarteslabs as dl
from descarteslabs.client.services import Places
from tqdm import tqdm
import sys
from pyproj import Proj, transform
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
sys.path.append('../util')
from constants import *

def get_tile_lat_lon(tile):
    """Return the lat, lon coordinates of a DLTile"""
    min_x, min_y, max_x, max_y = tile.bounds
    center_x, center_y = (max_x + min_x)/2, (max_y + min_y)/2
    lat, lon = transform(Proj(tile.proj4), LAT_LON_EPSG, center_x, center_y)
    return lat, lon

if __name__ == "__main__":
    tilesize = int(HANSEN_TILE_SIZE_KM * 1000 / NLCD_PRODUCT_RES)
    pad = 0
    year = 2015 #set back one year because download_images adds one year
    us_shape = 'north-america_united-states'
    startdate = '2016-01-01'
    enddate = '2017-01-01'
    label_header = 'label_path'
    keep_tiles = []
    landsat_product = [LANDSAT8_PRE_COLLECTION_PRODUCT_NAME, LANDSAT8_TIER1_PRODUCT_NAME]
    cols = [LATITUDE_HEADER, LONGITUDE_HEADER, YEAR_HEADER, label_header]
    data = defaultdict(list)
    print("Initalized variables")
    
    train_lat_lon_df = pd.read_csv(HANSEN_TRAIN_CURTIS_PATH, header=None)
    test_lat_lon_df = pd.read_csv(HANSEN_TEST_CURTIS_PATH, header=None)
    val_lat_lon_df = pd.read_csv(HANSEN_VAL_CURTIS_PATH, header=None)
    
    header_names = train_lat_lon_df.iloc[0]
    test_lat_lon_df = test_lat_lon_df.iloc[1:]
    val_lat_lon_df = val_lat_lon_df.iloc[1:]
    train_lat_lon_df = train_lat_lon_df.iloc[1:]
    all_lat_lon_df = train_lat_lon_df.append([test_lat_lon_df,val_lat_lon_df])
    all_lat_lon_df.columns = header_names
    
    dataset_tiles = [
            dl.scenes.DLTile.from_latlon(
            lat=float(row.latitude), lon=float(row.longitude),
            resolution=LANDSAT8_TIER1_PRODUCT_RES,
            tilesize=2, pad=int(tile_size / 2) - 1 #not sure about the tilesize or padding
        )
        for _, row in all_lat_lon_df.iterrows()
    ]  
    print("Created list of Hansen tiles")
    
    us = Places().shape(us_shape)
    ustiles = dl.scenes.DLTile.from_shape(us, NLCD_PRODUCT_RES, tilesize, pad)
    print("Created us tile shape")
    
    for tile in tqdm(ustiles):
        if any(tile.geometry.intersects(dataset_tile.geometry) for dataset_tile in dataset_tiles):
            continue
        scenes,ctx = dl.scenes.search(aoi=tile,products=landsat_product,cloud_fraction=0.01,limit=1)
        if len(scenes) > 0:
            keep_tiles.append(tile)
    print("Finished creating list of tiles to keep")
    
    for tile in tqdm(keep_tiles):
        scenes,ctx = dl.scenes.search(aoi=tile,products=NLCD_PRODUCT_NAME,start_datetime=startdate, end_datetime=enddate, limit=1)
        if len(scenes) > 0:
            lat, lon = get_tile_lat_lon(tile)
            mask_path = NLCD_LABELS_PATH / f'{lat}_{lon}.npy'
            data[LATITUDE_HEADER].append(lat)
            data[LONGITUDE_HEADER].append(lon)
            data[YEAR_HEADER].append(year)
            data[label_header].append(mask_path)
            mask = scenes[0].ndarray("land_cover", ctx, mask_nodata=True).data
            np.save(mask_path, mask)
            
    print("Finished downloading all labels")
    df = pd.DataFrame(data=data,columns=cols)
    
    train_path = NLCD_PATH / "train_latlon_landcover.csv"
    val_path = NLCD_PATH / "val_latlon_landcover.csv"
    test_path = NLCD_PATH / "test_latlon_landcover.csv"

    trainval_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )
    train_df, val_df = train_test_split(
        trainval_df, test_size=0.2, random_state=42
    )

    print("Writing dataframe with paths to:")
    print(train_path)
    print(val_path)
    print(test_path)

    train_df.to_csv(train_path, index=False, header=False)
    val_df.to_csv(val_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=False)
