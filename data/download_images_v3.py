from argparse import ArgumentParser
import os
import descarteslabs as dl
import pandas as pd
import sys
import fire
from tqdm import tqdm
import numpy as np
from PIL import Image
sys.path.append('../util')
from constants import *
from data_util import * #data_utils
import pdb
import concurrent.futures
import time
import datetime
import argparse

# NOTE: passing in dates_fn and products_fn methods allows us to still 
# use TileDownloader for the Curtis et al. dataset if necessary 
class TileDownloader():
    def __init__(self, dataset, cloud_fraction_fn, scene_limit, dates_fn, products_fn):
        """Initialize the TileDownloader object, which is able to download
        tiles given a (lat, lon) coordinate.

        Args:
            dataset           (str):     hansen
            cloud_fraction_fn  (float):  Function that sets the cloud fraction to None (for pre-LS8)
                                          or [0, 1] for LS8. Return scenes where % of clouds is
                                         less than this number (between 0 and 1)
            scene_limit       (int):     Max num scenes to pull from DL
            dates_fn           (fn):     Function that returns start_datetime, end_datetime
                                         for dl scene search. (year) passed as param.
            products_fn        (fn):     Function that returns products for dl scene search.
                                         (year) passed as param.                                
        """
        self._dataset = dataset 
        self._cloud_fraction_fn = cloud_fraction_fn
        self._scene_limit = scene_limit
        self.tile_size_km = HANSEN_TILE_SIZE_KM
        self._resolution = LANDSAT8_TIER1_PRODUCT_RES
        self._dates_fn = dates_fn 
        self._products_fn = products_fn

    def get_sc_from_latlon(self, lat, lon, year, multi_year=False):
        """Search DL for a tile with the specified (lat, lon)
        as its centroid, return the first scene found.
        """
        north_hs = lat >= 0
        start_time, end_time = self._dates_fn(year, north_hs, multi_year)
        products = self._products_fn(year+1, multi_year) # Always look at least a year ahead
        cloud_fraction = self._cloud_fraction_fn(products)
        cloud_bands_used = not (cloud_fraction is None)
        # Size of the tile (in px) is a function of resolution and
        # fact that we want 10x10km tiles
        original_tile_size = int(self.tile_size_km * 1000 / self._resolution) #666

        pad = int(original_tile_size / 2) - 1 #332
        
        # Since the position of (lat, lon) may vary within the tile
        # in .from_latlon, we set tilesize=2 and use padding to force
        # (lat, lon) to be the center of the tile.
        print(f"\t Date range queried: ({start_time},{end_time}), event year: {year}, products: {products}, \
            cloud_fraction: {cloud_fraction}")

        tile = dl.scenes.DLTile.from_latlon(lat=lat, lon=lon,
                                            resolution=self._resolution,
                                            tilesize=HANSEN_TILE_SIZE_PX,
                                            pad=pad)

        scenes, ctx = dl.scenes.search(aoi=tile,
                                       products=products,
                                       start_datetime=start_time,
                                       end_datetime=end_time,
                                       cloud_fraction=cloud_fraction,
                                       limit=self._scene_limit,
                                       sort_field="acquired",
                                       sort_order="desc")

        return scenes, ctx, products, cloud_bands_used
    
    def get_band(self, sc_stack, band):
        if band == 'rgb':
            return sc_stack[:, BANDS.index(RGB_BANDS[0]):BANDS.index(RGB_BANDS[-1])+1, :, :]
        elif band == 'ir':
            return sc_stack[:, BANDS.index(IR_BANDS[0]):BANDS.index(IR_BANDS[-1])+1, :, :]        
        return sc_stack[:, BANDS.index(band):BANDS.index(band)+1, :, :]

    def get_cloud_mask(self, sc_stack):
        cloud_mask = self.get_band(sc_stack, CLOUD_MASK_BAND)
        cloud_mask = np.tile(cloud_mask, (1, 3, 1, 1))
        # | the bright-mask to the cloud-mask for more conservative
        # cloud filtering
        bright_mask = self.get_band(sc_stack, BRIGHT_MASK_BAND)
        bright_mask = np.tile(bright_mask, (1, 3, 1, 1))
        return cloud_mask | bright_mask 

    def get_missing_px_mask(self, sc_stack):
        # for 'cut' images
        rgb = self.get_band(sc_stack, 'rgb')
        return rgb.mask.astype(np.uint8)

    def sort_scenes_by_num_px_masked(self, sc_stack):
        cloud_mask = self.get_cloud_mask(sc_stack)
        missing_px_mask = self.get_missing_px_mask(sc_stack)
        mask = cloud_mask.data | missing_px_mask 
        num_px_masked = np.sum(mask, axis=(1, 2, 3))
        sorted_idxs = np.argsort(num_px_masked)
        sc_stack_sorted = sc_stack[sorted_idxs]
        mask_sorted = mask[sorted_idxs]
        return sc_stack_sorted, mask_sorted

    # The images we are looking for mett:
    # (% cloudy px < SINGLE_IMG_CLOUD_FRAC) U (# cirrus band px == 0)
    def get_single_image(self, sc_stack_sorted, mask_sorted):
        img_size = sc_stack_sorted[0].size
        cirrus = self.get_band(sc_stack_sorted, CIRRUS_BAND)    
        num_px_masked = np.sum(mask_sorted, axis=(1, 2, 3))    
        num_px_cirrus = np.sum(cirrus, axis=(1, 2, 3))
        low_cloud_mask_img_idxs = np.where(num_px_masked < SINGLE_IMG_CLOUD_FRAC * img_size)[0]
        low_cirrus_img_idxs = np.where(num_px_cirrus == 0)[0]
        low_cloud_img_idxs = np.intersect1d(low_cloud_mask_img_idxs, low_cirrus_img_idxs)
        for idx in low_cloud_img_idxs:
            rgb_img = self.get_band(sc_stack_sorted, 'rgb')[idx]
            if np.sum(rgb_img.mask.astype(np.uint8)) > 0:
                # this image has missing pixels, skip it 
                continue 
            else:
                ir_img = self.get_band(sc_stack_sorted, 'ir')[idx]
                return rgb_img, ir_img 
        return None 

    def get_composite_img(self, rgb_stack, mask):
        rgb_stack_masked = np.ma.MaskedArray(rgb_stack, mask=mask)
        composite = np.ma.median(rgb_stack_masked, axis=0)
        return composite

    def get_small_composite(self, sc_stack_sorted, mask_sorted):
        # we only care about first SMALL_COMP_SC_NUM scenes here
        sc_stack_sorted = sc_stack_sorted[:SMALL_COMP_SC_NUM]
        mask_sorted = mask_sorted[:SMALL_COMP_SC_NUM]
        if mask_sorted.sum() < mask_sorted.size * SMALL_COMP_CLOUD_FRAC:
            rgb_composite = self.get_composite_img(self.get_band(sc_stack_sorted, 'rgb'), mask_sorted)
            ir_composite = self.get_composite_img(self.get_band(sc_stack_sorted, 'ir'), mask_sorted)
            return rgb_composite, ir_composite 
        return None 

    def get_full_composite(self, sc_stack_sorted, mask_sorted):
        rgb_composite = self.get_composite_img(self.get_band(sc_stack_sorted, 'rgb'), mask_sorted)        
        ir_composite = self.get_composite_img(self.get_band(sc_stack_sorted, 'ir'), mask_sorted)        
        return rgb_composite, ir_composite

    def download_masked(self, scenes, ctx, rgb_path, ir_path, cloud_bands = False):
        """Download RGB bands of tile to specified image_path.
        We first try to retrieve a single high-quality image from the tile stack.
        If this doesn't work, we try a small tile stack composite, and at the last resort
        fall back on a full tile stack image composite. 
        """
        if cloud_bands:
            bands = RGB_BANDS + IR_BANDS + CLOUD_BANDS
        else:
            bands = RGB_BANDS + IR_BANDS
        sc_stack = scenes.stack(bands,
                                ctx,
                                scaling="display",
                                processing_level="surface")
        sc_stack_sorted, mask_sorted = self.sort_scenes_by_num_px_masked(sc_stack)
        
        imgs = self.get_single_image(sc_stack_sorted, mask_sorted)
        if imgs is not None:
            download_method = SINGLE_IMG_DOWNLOAD_METHOD
        else:
            imgs = self.get_small_composite(sc_stack_sorted, mask_sorted)
            if imgs is not None:
                download_method = SMALL_COMPOSITE_DOWNLOAD_METHOD
            else:
                imgs = self.get_full_composite(sc_stack_sorted, mask_sorted)
                download_method = FULL_COMPOSITE_DOWNLOAD_METHOD

        rgb, ir = imgs 

        rgb = np.transpose(np.uint8(rgb.data), (1, 2, 0))
        rgb = Image.fromarray(rgb)
        rgb.save(rgb_path)  
        
        if ir_path is not None:
            ir = np.transpose(np.uint16(ir.data), (1, 2, 0)) 
            np.save(ir_path, ir)
            
        return download_method 


    def download_tile(self, lat, lon, year, rgb_path, ir_path, multi_year=False):
        scenes, ctx, _, cloud_bands_used = self.get_sc_from_latlon(lat, lon, year, multi_year)
        num_scenes = len(scenes)
        method = None
        if num_scenes > 0:
            download_method = self.download_masked(scenes, ctx, rgb_path, ir_path, cloud_bands_used)
        else:
            download_method = None
        return num_scenes, download_method


def download_event(td,
                   row,
                   index,
                   start_idx,
                   use_ir,
                   download_multiyear,
                   debug_log
    ):
    
    # row = metadata[index]
    if index < start_idx:
        print(f"Index {index} less than starting index {start_idx}. Skipping ...")
        return [], index
        
    year = row[YEAR_HEADER]

    if year + 1 < CUTOFF_YEAR:
        print(f"Skipping pre-{CUTOFF_YEAR} event {index}, year {year}")
        return [], index
        
    lat, lon = row[LATITUDE_HEADER], row[LONGITUDE_HEADER]

    parent_dir, fname = os.path.split(image_folder)
    fname += f'_{round(lat, 5)}_{round(lon, 5)}'
    image_folder = os.path.join(parent_dir, fname)
    print("Saving to ", image_folder)
    
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    event_sc, event_download = [], [] #Num scenes and download methods for one event
    event_imgs = 0
            
    for t in range(TEMPORAL_RANGE):
        if year+t+1 > PRESENT_YEAR: 
            event_imgs += 0 #Cannot pull data from future
            continue 

        image_filename = f'{int(year)+t}.png'

        image_path = os.path.join(image_folder, image_filename)

        if use_ir:
            ir_filename = f'{int(year)+t}.npy'
            ir_path = os.path.join(image_folder, ir_filename)
        else:
            ir_path = None

        try:
            num_sc, download_method = td.download_tile(lat, lon, year+t, image_path, ir_path)
        except Exception as e:
            error_msg = f"Error {e} for event idx {index}, loss year {year}, year index {t}"
            print(error_msg)
            debug_log.write(error_msg + '\n')
            num_sc = 0
            download_method = None

        event_sc.append(str(num_sc))
        if download_method is not None:
            event_download.append(download_method)
        else:
            event_download.append('None')

        if num_sc > 0:
            event_imgs += 1

    ## Download multi-year composite.
    if download_multiyear:

        multiyear_image_filename = f'multi_{year+1}_{year+TEMPORAL_RANGE}.png'
        multiyear_image_path = os.path.join(image_folder, multiyear_image_filename)

        if use_ir:
            multiyear_ir_filename = f'multi_{year+1}_{year+TEMPORAL_RANGE}.npy'
            multiyear_ir_path = os.path.join(image_folder, multiyear_ir_filename)
        else:
            multiyear_ir_path = None

        try:
            num_sc, download_method = td.download_tile(lat, lon, year, multiyear_image_path, multiyear_ir_path, \
                multi_year=True)
        except Exception as e:
            error_msg = f"Error {e} in multiyear composite for event idx {index}, loss year {year}, year index {t}"
            print(error_msg)
            debug_log.write(error_msg + '\n')


    print(f'\t Scenes acquired for event {index}: {event_sc}')
            
    if event_imgs > 0:
        print(f'\t Downloaded {event_imgs} images for event {index}')
        event_data = [row[LABEL_HEADER], 
                      lat,
                      lon,
                      year,
                      image_folder,
                      event_imgs,
                      '-'.join(event_sc),
                      '-'.join(event_download),
                      multiyear_image_path, 
                      multiyear_ir_path
                     ]

    else:
        print(f'\t Found no images for event {index}')
        print(f'Removing folder {image_folder}')
        try:
            os.rmdir(image_folder)
        except FileNotFoundError:
            raise Exception("Folder is non-empty but has no images!")
        event_data = []

    return event_data, index

def download_images_parallel(dataset,
                    meta_filename, 
                    log_path,
                    cloud_fraction_fn,
                    scene_limit=SCENE_LIMIT,
                    use_ir=True, 
                    download_multiyear=True,
                    save_every=20,
                    start_idx=0,
                    max_workers=8,
                    debug_log_dir='logs/'
                   ):
    """Download mosaics of Landsat images described in the metapath. 
    Maintains and saves a log file containing information derived metadata about the images download  
    By default uses multithreading  for downloads. Specify num_workers=1 for single thread execution.

    Args:
        dataset           (str):    'hansen'
        dates_fn           (fn):    Function that returns start_datetime, end_datetime
                                    for dl scene search. (year) passed as param.
        products_fn        (fn):    Function that returns products for dl scene search.
                                    (year) passed as param.   
        cloud_fraction_fn  (float): Function to set cloud fraction based on product.
                                    None if LS7 prod used, else CLOUD_FRACTION (return scenes where % of clouds is
                                    less than this number (between 0 and 1))
        scene_limit       (int):    Max num scenes to pull from DL
        use_ir           (bool):    True if IR bands used, false otherwise
        download_multiyear(bool):   Download a multiyear composite
        save_every         (int):   Interval of saving to log path
        start_idx          (int):   Ignore indices before this one
        num_workers        (int):   Max number of processes to use for download
        
    """

    print(f"Downloading from metafile at {meta_filename}")

    assert(os.path.exists(meta_filename))
    metadata = pd.read_csv(meta_filename,
                           header=None,
                           names=HANSEN_DOWNLOAD_META_COLNAMES)

    td = TileDownloader(dataset,
                        cloud_fraction_fn,
                        scene_limit,
                        hansen_dates_fn,
                        hansen_products_fn)
    indices_list = []
    data_list = [] #Main 2D data list
    start_time = time.time()

    now = datetime.datetime.now()

    if not os.path.exists(debug_log_dir):
        os.makedirs(debug_log_dir)

    debug_log = open(os.path.join(debug_log_dir, f'logs_{now.strftime("%m/%d/%Y, %H:%M:%S")}'))
    print(f'Using {max_workers} to parallelize...')

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        downloaded_events = {executor.submit(download_event,
                                                td,
                                                row,
                                                index, 
                                                start_idx,
                                                use_ir,
                                                download_multiyear,
                                                debug_log): (index, row) for index, row in metadata.iterrows()}
        found = 0 
        num_not_found = 0

        for future in tqdm(concurrent.futures.as_completed(downloaded_events), total=metadata.shape[0]):
            event_data, index = future.result()
            if len(event_data) == 0:
                num_not_found += 1
            else:
                data_list.append(event_data)
                indices_list.append(index)
                found += 1
                if found % save_every == 0:
                    new_metadata = pd.DataFrame(data_list, index=indices_list, \
                        columns = HANSEN_ORIGINAL_META_COLNAMES)
                    print(f"Downloaded {len(indices_list)} out of {len(metadata)} events")
                    print(f"Saving post-download metadata to {log_path}")
                    new_metadata.to_csv(log_path, header=True, index=True)

            download_time = time.time() - start_time
            print(f'{download_time}s for {found} events')

    new_metadata = pd.DataFrame(data_list, index=indices_list, columns = HANSEN_ORIGINAL_META_COLNAMES)
    print(f"Downloaded {len(indices_list)} out of {len(metadata)} events")
    print(f"Saving post-download metadata to {log_path}")
    new_metadata.to_csv(log_path, header=True, index=True)
    debug_log.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="Split of data to download", \
        type=str, choices=('train', 'val', 'test'))
    args = parser.parse_args()

    if not args.split:
        raise Exception("Must pass in data split to download!")
    split = args.split

    print(f'Downloading {split} split')
    download_images_parallel('hansen', 
        DATA_BASE_DIR/f'predownload_meta_{split}.csv',
        DATA_BASE_DIR/f'postdownload_meta_{split}_v3.csv',
        cloud_fraction_fn=hansen_cloud_fractions_fn,
        scene_limit=SCENE_LIMIT,
        use_ir=True)
