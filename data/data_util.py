import sys
sys.path.append('../util')
from constants import *
import pandas as pd
from PIL import Image
import numpy as np
from datetime import datetime



def partition_sc_by_yr(scenes):
    """
    Take in list of date strings. No particular ordering assumed
    Output dict mapping year to set of idxs
    """
    year_map = dict()
    for idx, sc in enumerate(scenes):
        year = sc.properties.date.strftime("%Y")
        if year in year_map:
            year_map[year].append(idx)
        else:
            year_map[year] = [idx]

    return year_map
        
def yearwise_counts(years, counts, reference, T = 4):
    """
    Return a list of counts for events in each of the T years from the reference year (exclusive)
    Current assumption is that length of years <= T, though this can change in future.
    
    Args: 
        years (list<int>) : list of years
        counts (list<int>): number of scenes for each year in years
        reference (int)   : year from which counts are considered
        T (int)           : length of temporal window over which we track cts
    Returns:
        list<int> of length T, each value ct for that year after reference
    """
    years = sorted(years) #make sure years sorted
    year_cts = [0 for _ in range(T)]
    
    for i, yr in enumerate(years):
        idx = yr - reference - 1 #indices zero-centered
        if idx >= 0 and idx < T: year_cts[idx] += counts[i]
    return year_cts


############# Functions below imported from Aut 19 Indonesia repo#################################
##https://github.com/stanfordmlgroup/aicc-aut19-deforestation/blob/master/data/download_util.py##
def get_dates(year, landsat8, years_following=4):
    if landsat8 and year < 2012:
        year = 2012
    return f'{year + 1}-01-01', f'{year + years_following + 1}-01-01'


def get_products(landsat, use_sentinel):
    if use_sentinel:
        return [SENTINEL2_PRODUCT_NAME]
    elif landsat == 8:
        return [LANDSAT8_PRE_COLLECTION_PRODUCT_NAME,
                LANDSAT8_TIER1_PRODUCT_NAME]
    elif landsat == 7:
        return [LANDSAT7_PRE_COLLECTION_PRODUCT_NAME]
    elif landsat == 5:
        return [LANDSAT5_PRE_COLLECTION_PRODUCT_NAME]
    else:
        raise ValueError(f"Landsat {landsat} not supported.")


def get_bands(is_landsat8, download_ir):
    if is_landsat8:
        if download_ir:
            return BANDS_LS8
        else:
            return BANDS_LS8_NO_IR
    else:
        if download_ir:
            return BANDS_LS7
        else:
            return BANDS_LS7_NO_IR


def get_band(sc_stack, band, is_landsat8, download_ir):
    bands = get_bands(is_landsat8, download_ir)

    if band == 'rgb':
        start_index = bands.index(RGB_BANDS[0])
        end_index = bands.index(RGB_BANDS[-1])+1

    elif band == 'ir':
        start_index = bands.index(IR_BANDS[0])
        end_index = bands.index(IR_BANDS[-1])+1

    else:
        start_index = bands.index(band)
        end_index = bands.index(band)+1

    return sc_stack[:, start_index:end_index, :, :]


def get_cloud_mask(sc_stack, is_landsat8, download_ir):
    if is_landsat8:
        cloud_mask = get_band(sc_stack, CLOUD_MASK_BAND,
                              is_landsat8=is_landsat8,
                              download_ir=download_ir)
        cloud_mask = np.tile(cloud_mask, (1, 3, 1, 1))
        # | the bright-mask to the cloud-mask for more conservative
        # cloud filtering
        bright_mask = get_band(sc_stack, BRIGHT_MASK_BAND,
                               is_landsat8=is_landsat8,
                               download_ir=download_ir)
        bright_mask = np.tile(bright_mask, (1, 3, 1, 1))
        cloud_mask = cloud_mask | bright_mask
    else:
        cloud_mask = get_band(sc_stack, CLOUD_MASK_BAND_LS7,
                              is_landsat8=is_landsat8,
                              download_ir=download_ir)
        cloud_mask = np.tile(cloud_mask, (1, 3, 1, 1))

    return cloud_mask


def get_missing_px_mask(sc_stack, is_landsat8, download_ir):
    # for 'cut' images
    rgb = get_band(sc_stack, 'rgb', is_landsat8, download_ir)
    return rgb.mask.astype(np.uint8)


def get_mask(sc_stack, is_landsat8, download_ir):
    cloud_mask = get_cloud_mask(sc_stack, is_landsat8, download_ir)
    missing_px_mask = get_missing_px_mask(sc_stack, is_landsat8, download_ir)
    mask = cloud_mask | missing_px_mask

    return mask


def find_low_cloud_ls8_scenes(sc_stack, download_ir, n=None):

    mask = get_mask(sc_stack, is_landsat8=True, download_ir=download_ir)
    num_px_masked = np.sum(mask, axis=(1, 2, 3))

    img_size = sc_stack[0].size
    low_cloud_mask_img_idxs = np.where(
        num_px_masked < SINGLE_IMG_CLOUD_FRAC * img_size
    )[0]
    # Get the n minimum scenes if less than n scenes were found
    if n is not None and len(low_cloud_mask_img_idxs) < n:
        low_cloud_mask_img_idxs = num_px_masked.argsort()[:n]

    return low_cloud_mask_img_idxs

def find_low_cirrus_ls8_scenes(sc_stack, download_ir):
    cirrus = get_band(sc_stack, CIRRUS_BAND, is_landsat8=True,
                      download_ir=download_ir)
    num_px_cirrus = np.sum(cirrus, axis=(1, 2, 3))

    low_cirrus_img_idxs = np.where(num_px_cirrus == 0)[0]

    return low_cirrus_img_idxs

def find_low_cloud_ls7_scenes(sc_stack, download_ir):
    cloud_mask = get_cloud_mask(sc_stack, is_landsat8=False,
                                download_ir=download_ir)
    num_px_cloud = np.sum(cloud_mask, axis=(1, 2, 3))

    img_size = sc_stack[0].size
    low_cloud_mask_img_idxs = np.where(
        num_px_cloud < SINGLE_IMG_CLOUD_FRAC_LS7 * img_size
    )[0]

    return low_cloud_mask_img_idxs

def find_high_ndvi_ls7_scenes(sc_stack, download_ir):
    ndvi = get_band(sc_stack, NDVI_BAND_LS7, is_landsat8=False,
                    download_ir=download_ir)
    ndvi_mean = np.mean(ndvi, axis=(1, 2, 3))
    high_ndvi_img_idxs = np.where(
        ndvi_mean > NDVI_IMG_MEAN_LS7
    )

    return high_ndvi_img_idxs

def find_low_cloud_high_ndvi_ls7_scenes(sc_stack, download_ir):
    low_cloud_mask_img_idxs = find_low_cloud_ls7_scenes(sc_stack, download_ir)
    high_ndvi_img_idxs = find_high_ndvi_ls7_scenes(sc_stack, download_ir)

    low_cloud_img_idxs = np.intersect1d(
        low_cloud_mask_img_idxs, high_ndvi_img_idxs
    )

    return low_cloud_img_idxs


def stack_to_median_composite(band_stack, mask_stack):
    band_stack_masked = np.ma.MaskedArray(band_stack, mask=mask_stack)
    composite = np.ma.median(band_stack_masked, axis=0)
    return composite


def numpy_to_pil(arr):
    arr_transpose = np.transpose(np.uint8(arr.data), (1, 2, 0))
    img = Image.fromarray(arr_transpose)
    return img


def prep_ir(arr):
    return np.transpose(np.uint16(arr.data), (1, 2, 0))


def get_paths(single_rgb_imgs):
    rgb_paths = [rgb_path for _, rgb_path in single_rgb_imgs]

    cloud_pxs = [int(rgb_path.split("cloud_")[1].split(".")[0])
                    for rgb_path in rgb_paths]

    str_dates = [rgb_path.split("_cloud")[0].split("/")[1]
                    for rgb_path in rgb_paths]
    dates = [datetime.strptime(str_date, "%Y_%m_%d")
                for str_date in str_dates]

    least_cloudy_im_path = rgb_paths[np.argmin(cloud_pxs)]
    closest_date_im_path = rgb_paths[np.argmin(dates)]
    furthest_date_im_path = rgb_paths[np.argmax(dates)]

    return least_cloudy_im_path, closest_date_im_path, furthest_date_im_path


###################################################################################


###################3 HANSEN v3 UTILS #############################
def hansen_dates_fn(year, hemisphere, multi_year=False, whole_year=True):
    """
    Return the start and end dates for the given year and hemisphere.
    By default, ignores hemispheres and downloads images for the whole year following the image
    
    Args: 
        year (int): year of forest loss
        hemisphere (str): {'north', 'south'} indicating the hemisphere of event 
    Output:
        start (str): start date
        end (str): end date
    """

    ## Generate multiyear composite
    if multi_year:
        start = f'{year + 1}-01-01'
        end = f'{year + 4}-12-31' 

    else:
        if whole_year:
            start = f'{year + 1}-01-01'
            end = f'{year + 1}-12-31'

        else:
            # Summer months assumed to be [Jun... Aug] in N hemisphere and [Dec... Feb] in S
            # The ranges for winter switch between the two hemispheres
            if hemisphere == 'north':
                start = f'{year + 1}-06-01' 
                end = f'{year + 1}-08-31'
            else:
                start = f'{year + 1}-12-01'
                end = f'{year + 2}-02-01'

    return start, end

def hansen_products_fn(year, multi_year=False):
    if multi_year:
        products = [LANDSAT8_PRE_COLLECTION_PRODUCT_NAME, LANDSAT8_TIER1_PRODUCT_NAME]
                    
    else:
        if year >= 2013:
            products = [LANDSAT8_PRE_COLLECTION_PRODUCT_NAME, LANDSAT8_TIER1_PRODUCT_NAME]
        elif year == 2012:
            products = [LANDSAT8_PRE_COLLECTION_PRODUCT_NAME] 
        else:
            products = [LANDSAT7_PRE_COLLECTION_PRODUCT_NAME]
    return products 

def hansen_cloud_fractions_fn(products):
    if LANDSAT7_PRE_COLLECTION_PRODUCT_NAME in products:
        return None
    else:
        return CLOUD_FRACTION


def get_path(idx, save_dir):
    return save_dir / f'{idx}'.zfill(4)


def indo_to_hansen_download_meta(old_path, new_path, split, verbose=False):
    """
    Make necessary changes from train metadata csv from indonesia dataset to adapt it to Hansen data
    These include:
        i) Eliminating unnecessary columns
        ii) Editing image paths to point deforest-global instead of deforestation directory

    Inputs:
        old_path (str): Path to original csv in which to make changes 
        new_path (str): Path to original csv in which to make changes 
    Outputs:
        None
    """
    save_dir = DATA_BASE_DIR / f'images_v3/{split}/'

    print(f"Old path {old_path}, new path {new_path}")
    df_old = pd.read_csv(old_path, header=None, names=None)
    num_cols = len(df_old.columns)
    num_cols_to_keep = 4 
    kept_idxs = list(range(num_cols_to_keep)) + [num_cols - 1]
    dropped_idxs = [idx for idx in range(num_cols) if idx not in kept_idxs]
    print(f"Dropping cols {dropped_idxs}")

    df_old = df_old.drop(columns=dropped_idxs)
    path_col = len(df_old.columns) - 2 

    # some paths contain Nones, so reset indices
    df_old[path_col] = df_old.apply(lambda row: get_path(row.name, save_dir), axis=1)

    df_old.to_csv(new_path, header=False, index=False)



    ######################### HANSEN V4 UTILS