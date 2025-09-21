import os
import sys
import fire
import json
import traceback
import numpy as np
import pandas as pd
import descarteslabs as dl
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, '../')
from util.constants import *
from data_util import *
import warnings
import copy
import argparse

warnings.filterwarnings("ignore", category=DeprecationWarning)

MAX_YRS_PER_EVENT = 4

class TileDownloader():
    def __init__(self, cloud_fraction, scene_limit,
                 only_closest, download_ir,
                 years_following, center_tiles,
                 download_sentinel):
        """Initialize the TileDownloader object, which is able to download
        tiles given a (lat, lon) coordinate.

        Args:
            cloud_fraction  (float):  Return scenes where % of clouds is
                                      less than this number (between 0 and 1)
            scene_limit       (int):  Max num scenes to pull from DL
            only_closest     (bool):  Whether to only download the image
                                      closest to the event.
            download_ir      (bool):  Whether to download IR images.
            years_following   (int):  Number of years following the event
                                      to search for scenes.
            center_tiles     (bool):  Whether to center tiles around the
                                      lat lons.
            download_sentinel (bool): Whether to download sentinel 2 instead of
                                      landsat.
        """
        self._cloud_fraction = cloud_fraction
        self._scene_limit = scene_limit
        self._tile_size_km = HANSEN_TILE_SIZE_KM
        if not download_sentinel:
            self._resolution = LANDSAT8_TIER1_PRODUCT_RES
        else:
            raise Exception("Sentinel product undefined!")
            # self._resolution = SENTINEL2_PRODUCT_RES
        self._only_closest = only_closest
        self._download_ir = download_ir
        self._years_following = years_following
        self._center_tiles = center_tiles
        self._download_sentinel = download_sentinel

    def get_sc_from_latlon(self, lat, lon, year, landsat):
        """Search DL for a tile with the specified (lat, lon)
        as its centroid, return the first scene found.
        """
        is_landsat8 = landsat == 8
        start_time, end_time = get_dates(year, is_landsat8,
                                         self._years_following)
        products = get_products(landsat, self._download_sentinel)

        # Size of the tile (in px) is a function of resolution and
        # fact that we want 10x10km tiles
        tile_size = int(self._tile_size_km * 1000 / self._resolution)

        if self._center_tiles:
            # Since the position of (lat, lon) may vary within the tile
            # in .from_latlon, we set tilesize=2 and use padding to force
            # (lat, lon) to be the center of the tile.
            pad = int(tile_size / 2) - 1
            tile_size = 2
        else:
            pad = 0
        tile = dl.scenes.DLTile.from_latlon(lat=lat, lon=lon,
                                            resolution=self._resolution,
                                            tilesize=tile_size,
                                            pad=pad)
        cloud_fraction = self._cloud_fraction if is_landsat8 else None
        scenes, ctx = dl.scenes.search(aoi=tile,
                                       products=products,
                                       cloud_fraction=cloud_fraction,
                                       start_datetime=start_time,
                                       end_datetime=end_time,
                                       limit=self._scene_limit,
                                       sort_field="acquired",
                                       sort_order="desc")
        return scenes, ctx, products

    def create_single_images(
            self, sc_stack, scenes,
            low_cloud_img_idxs,
            is_landsat8
        ):

        mask = get_mask(sc_stack, is_landsat8=is_landsat8,
                        download_ir=self._download_ir)
        num_px_masked = np.sum(mask, axis=(1, 2, 3))

        rgb_imgs, ir_imgs = [], []
        rgb_sc_stack = get_band(sc_stack, 'rgb', is_landsat8=is_landsat8,
                                download_ir=self._download_ir)
        if self._download_ir:
            ir_sc_stack = get_band(sc_stack, 'ir', is_landsat8=is_landsat8,
                                   download_ir=self._download_ir)
        for idx in low_cloud_img_idxs:
            rgb_sc = rgb_sc_stack[idx]
            if np.sum(rgb_sc.mask.astype(np.uint8)) > 0:
                # this image has missing pixels, skip it
                continue
            rgb_img = numpy_to_pil(rgb_sc)

            scene = scenes[idx]
            img_date = scene.properties.date.strftime("%Y_%m_%d")
            filename = f'{img_date}_cloud_{num_px_masked[idx]}'
            rgb_path = f'rgb/{filename}.png'
            
            rgb_imgs.append((rgb_img, rgb_path))

            if self._download_ir:
                ir_sc = ir_sc_stack[idx]
                ir_img = prep_ir(ir_sc)
                ir_path = f'ir/{filename}.npy'
                ir_imgs.append((ir_img, ir_path))

        return rgb_imgs, ir_imgs

    def get_single_scenes(self, sc_stack, scenes, is_landsat8):
        """Get individual Landsat scenes.
        
        Return PIL images of accepted scenes.
        """
        if is_landsat8:
            # Get Landsat 8 scenes with a low amount of clouds
            # using the native cloud, brightness, and cirrus bands.
            low_cloud_mask_img_idxs = find_low_cloud_ls8_scenes(
                sc_stack, self._download_ir
            )
            low_cirrus_img_idxs = find_low_cirrus_ls8_scenes(
                sc_stack, self._download_ir
            )

            low_cloud_img_idxs = np.intersect1d(
                low_cloud_mask_img_idxs, low_cirrus_img_idxs
            )
        else:
            # Search for a single LS7 scene with low clouds
            # (ignoring artifacts since filtered later)
            # using cloud and ndvi (for filtering "foggy" images).
            low_cloud_img_idxs = find_low_cloud_high_ndvi_ls7_scenes(
                sc_stack, self._download_ir
            )

        if self._only_closest:
            # Keep only the closest cloud index if it exists
            if low_cloud_img_idxs.size != 0:
                dates = np.array([
                    scene.properties.date for scene in scenes
                ])
                low_cloud_img_idxs = np.array([
                    dates[low_cloud_img_idxs].argmin()
                ])

        rgb_imgs, ir_imgs = self.create_single_images(
            sc_stack, scenes, low_cloud_img_idxs, is_landsat8
        )

        return rgb_imgs, ir_imgs

    def get_composite(self, sc_stack, is_landsat8, year=None):

        if is_landsat8:
            composite_indices = find_low_cloud_ls8_scenes(
                sc_stack, download_ir=self._download_ir, n=5
            )
        else:
            composite_indices = find_low_cloud_high_ndvi_ls7_scenes(
                sc_stack, self._download_ir
            )
            num_scenes = len(composite_indices)
            mask = get_missing_px_mask(sc_stack, is_landsat8=is_landsat8, download_ir=True)
            mean_artifact_pxs = np.sum(mask, axis=(1, 2, 3)).mean()
            if ((num_scenes <= 3) or
                    (num_scenes <= 10 and mean_artifact_pxs > 100000)):
                return None, None

        mask = get_mask(sc_stack, is_landsat8=is_landsat8,
                        download_ir=self._download_ir)

        composite_sc_stack = sc_stack[composite_indices]
        composite_mask_stack = mask[composite_indices]

        # Cosntruct RGB composite
        composite_rgb_stack = get_band(composite_sc_stack, 'rgb',
                                       is_landsat8=is_landsat8,
                                       download_ir=self._download_ir)
        composite_rgb_arr = stack_to_median_composite(
            composite_rgb_stack, composite_mask_stack
        )
        composite_rgb_img = numpy_to_pil(composite_rgb_arr)
        
        rgb_tuple = (composite_rgb_img, "rgb/composite.png") if year == None \
        else (composite_rgb_img, f"rgb/{year}_annual.png") 

        if self._download_ir:
            # Construct IR composite
            composite_ir_stack = get_band(composite_sc_stack, 'ir',
                                          is_landsat8=is_landsat8,
                                          download_ir=self._download_ir)
            composite_ir_arr = stack_to_median_composite(
                composite_ir_stack,
                composite_mask_stack
            )
            composite_ir_img = prep_ir(composite_ir_arr)
            ir_tuple = (composite_ir_img, f"ir/composite.npy") if year == None \
            else (composite_rgb_img, f"ir/{year}_annual.npy") 
        else:
            ir_tuple = None

        return (rgb_tuple, ir_tuple)

    def write_images(self, single_rgb_imgs, single_ir_imgs,
                     composite_rgb_img, composite_ir_img,
                     annual_rgb_imgs, annual_ir_imgs,
                     images_path, is_landsat8
                    ):
        """
        Also generate annual composites if True
        """
        
        os.makedirs(images_path, exist_ok=True)

        # Get least cloudy and closest paths
        if len(single_rgb_imgs) == 0:
            least_cloudy_im_path = 'None'
            closest_date_im_path = 'None'
            furthest_date_im_path = 'None'
        else:
            least_cloudy_im_path, closest_date_im_path,\
            furthest_date_im_path = get_paths(single_rgb_imgs)

        # Create directories for single scenes
        os.makedirs(os.path.join(images_path, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(images_path, 'ir'), exist_ok=True)

        years_lst = []
        # Write all RGB images
        for rgb_img, rgb_path in single_rgb_imgs + annual_rgb_imgs + composite_rgb_img:
            rgb_img.save(os.path.join(images_path, rgb_path))
            if ('composite' not in rgb_path) and ('annual' not in rgb_path): 
                years_lst.append(int(rgb_path.split('/')[-1].split('_')[0]))

        # Write all IR images
        for ir_img, ir_path in single_ir_imgs + annual_ir_imgs + composite_ir_img:
            np.save(os.path.join(images_path, ir_path), ir_img)

        if len(composite_rgb_img) != 0:
            composite_im_path = composite_rgb_img[0][1]
        else:
            composite_im_path = 'None'

        download_metadata = {
            NUM_IMGS_DOWNLOADED: len(single_rgb_imgs) + len(composite_rgb_img),
            IMG_OPTION_CLOUD: least_cloudy_im_path,
            IMG_OPTION_CLOSEST_YEAR: closest_date_im_path,
            IMG_OPTION_FURTHEST_YEAR: furthest_date_im_path,
            IMG_OPTION_COMPOSITE: composite_im_path,
            IMG_COMPOSITE_IS_LS8: is_landsat8,
            IMG_PATH_HEADER: images_path,
        }
        
        return (download_metadata, years_lst)

    def download_ls8_images(self, lat, lon, year, images_path, annual_composites=True):
        """Download Landsat 8 images.

        Search for single Landsat 8 scenes with low amount of clouds
        using native cloud, brightness, and cirrus bands along
        with missing pixel mask. No guarantee these will be found.

        Always construct a composite dynamically by searching for scenes
        using the same procedure as for the single scene search, but ignoring
        the cirrus band. Take a masked median using the cloud and missing pixel
        mask.
        """
        
        scenes, ctx, _ = self.get_sc_from_latlon(lat, lon, year,
                                                 landsat=8)
        
        bands = get_bands(is_landsat8=True, download_ir=self._download_ir)
        sc_stack = scenes.stack(bands,
                                ctx,
                                scaling="display",
                                processing_level="surface")

        # Get single scenes
        single_rgb_imgs, single_ir_imgs =\
            self.get_single_scenes(sc_stack, scenes, is_landsat8=True)

        
        if not self._only_closest or len(single_rgb_imgs) == 0:
            # Get composite
            composite_rgb_img, composite_ir_img =\
                self.get_composite(sc_stack, is_landsat8=True)
            composite_rgb_img = [composite_rgb_img]
            composite_ir_img = (
                [composite_ir_img] if composite_ir_img is not None
                else []
            )
        else:
            composite_rgb_img = []
            composite_ir_img = []

        annual_rgb_imgs = []
        annual_ir_imgs = []
        
        download_annuals = (not self._only_closest) and (annual_composites)
        
        if download_annuals:
            ## Get annual idxs
            partitions = partition_sc_by_yr(scenes)
            
            ## Extract sc's from sc_stack for each yr
            stacks = {yr:sc_stack[partitions[yr]] for yr in partitions}
            print([(k, len(v)) for k, v in stacks.items()])

            ## generate annual composites
            single_sc_yrs = [yr for yr, sc_lst in partitions.items() if len(sc_lst) == 1]
            for yr, yearly_stack in stacks.items():
                yearly_rgb_img, yearly_ir_img =\
                    self.get_composite(yearly_stack, is_landsat8=True, year=yr)
                annual_rgb_imgs.append(yearly_rgb_img)
                if yearly_ir_img is not None:
                    annual_ir_imgs.append(yearly_ir_img)         

        download_metadata, years_lst = self.write_images(
            single_rgb_imgs, single_ir_imgs,
            composite_rgb_img, composite_ir_img,
            annual_rgb_imgs, annual_ir_imgs,
            images_path, is_landsat8=True
        )

        if download_annuals:      
            download_metadata['single_sc_annual_composites'] = '-'.join(single_sc_yrs)
        else:
            download_metadata['single_sc_annual_composites'] = 'N/A'
            
        unique, counts = np.unique(years_lst, return_counts = True)
        year_cts = yearwise_counts(unique, counts, int(year))
        
        return (download_metadata, year_cts)

    def get_ls5_or_ls7_scenes(self, lat, lon, year, landsat):
        scenes, ctx, _ = self.get_sc_from_latlon(lat, lon, year,
                                                 landsat=landsat)
        if len(scenes) == 0:
            return None, None
        bands = get_bands(is_landsat8=False, download_ir=self._download_ir)
        if self._download_ir:
            num_noncloud_bands = len(RGB_BANDS) + len(IR_BANDS)
        else:
            num_noncloud_bands = len(RGB_BANDS)
        sc_stack1 = scenes.stack(bands[:num_noncloud_bands],
                                 ctx,
                                 scaling="display",
                                 processing_level="surface")
        sc_stack2 = scenes.stack(bands[num_noncloud_bands:], ctx)
        sc_stack = np.ma.concatenate([sc_stack1, sc_stack2], axis=1)

        return sc_stack, scenes

    def download_ls7_images(self, lat, lon, year, images_path, annual_composites=True):
        
        sc_stack_ls7, scenes_ls7 = self.get_ls5_or_ls7_scenes(
            lat, lon, year, landsat=7
        )
        
        annual_scenes, annual_stack = scenes_ls7, sc_stack_ls7
        download_annuals = (not self._only_closest) and (annual_composites)
        
        # Get single scenes
        single_rgb_imgs, single_ir_imgs =\
            self.get_single_scenes(sc_stack_ls7, scenes_ls7, is_landsat8=False)

        if len(single_rgb_imgs) == 0:
            # If no single scene found, search for LS5
            sc_stack_ls5, scenes_ls5 = self.get_ls5_or_ls7_scenes(
                lat, lon, year, landsat=5
            )

            if sc_stack_ls5 is not None:
                sc_stack_ls7_ls5 = np.ma.concatenate([sc_stack_ls7,
                                                      sc_stack_ls5])
                scenes_ls7_ls5 = scenes_ls7
                scenes_ls7_ls5.extend(scenes_ls5)
                assert len(scenes_ls7_ls5) == len(sc_stack_ls7_ls5) # CRUCIAL there's a 1-1 map
                
                annual_stack, annual_scenes = sc_stack_ls7_ls5, scenes_ls7_ls5
            else:
                sc_stack_ls7_ls5 = sc_stack_ls7
                annual_stack, annual_scenes = sc_stack_ls7, scenes_ls7
                
                
            composite_rgb_img, composite_ir_img = self.get_composite(
                sc_stack_ls7_ls5, is_landsat8=False
            )

            if composite_rgb_img is None:
                # If no composite found, search for LS8
                return self.download_ls8_images(lat, lon, year, images_path)
        else:
            # Make "composite" image least cloudy single image
            # TODO: Consider changing this (e.g. composite over all single
            # images since we know none have artifacts, and this generalizes
            # single)
            least_cloudy_im_path, _, _ = get_paths(single_rgb_imgs)
            rgb_paths = [rgb_path for _, rgb_path in single_rgb_imgs]
            index = rgb_paths.index(least_cloudy_im_path)
            composite_rgb_img = (single_rgb_imgs[index][0], "rgb/composite.png")
            if self._download_ir:
                composite_ir_img = (single_ir_imgs[index][0], "ir/composite.npy")
            else:
                composite_ir_img = None

        composite_rgb_img_list = [composite_rgb_img]
        composite_ir_img_list = (
            [] if composite_ir_img is None
            else [composite_ir_img]
        )


        annual_rgb_imgs = []
        annual_ir_imgs = []
        
        if download_annuals:
            ## Get annual idxs
            partitions = partition_sc_by_yr(annual_scenes)
            ## Extract sc's from sc_stack for each yr
            stacks = {yr:annual_stack[partitions[yr]] for yr in partitions}
            print([(k, len(v)) for k, v in stacks.items()])

            ## generate annual composites
            single_sc_yrs = [yr for yr, sc_lst in partitions.items() if len(sc_lst) == 1]
            for yr, yearly_stack in stacks.items():
                yearly_rgb_img, yearly_ir_img =\
                    self.get_composite(yearly_stack, is_landsat8=True, year=yr)
                annual_rgb_imgs.append(yearly_rgb_img)
                if yearly_ir_img is not None:
                    annual_ir_imgs.append(yearly_ir_img)         

        download_metadata, years_lst = self.write_images(
            single_rgb_imgs, single_ir_imgs,
            composite_rgb_img_list, composite_ir_img_list,
            annual_rgb_imgs, annual_ir_imgs,
            images_path, is_landsat8=False
        )
        if download_annuals:      
            download_metadata['single_sc_annual_composites'] = '-'.join(single_sc_yrs)
        else:
            download_metadata['single_sc_annual_composites'] = 'N/A'
            
        unique, counts = np.unique(years_lst, return_counts = True)
        year_cts = yearwise_counts(unique, counts, int(year))

        return (download_metadata, year_cts)

    def download_images(self, lat, lon, year, images_path, download_ls7):
        if year < 2012 and not download_ls7:
            year = 2012

        if year >= 2012:
            return self.download_ls8_images(lat, lon, year, images_path)
        else:
            return self.download_ls7_images(lat, lon, year, images_path)

    def download_landcover(self, lat, lon, landcover_path):
        tile_size = int(self._tile_size_km * 1000 / self._resolution)
        dltile = dl.scenes.DLTile.from_latlon(
            lat=lat, lon=lon,
            resolution=self._resolution,
            tilesize=2, pad=int(tile_size / 2) - 1
        )
        scenes, ctx = dl.scenes.search(
            aoi=dltile, products=[INDONESIA_LANDCOVER_PRODUCT_NAME]
        )
        mask = scenes[0].ndarray("cover_class", ctx, mask_nodata=True).data
        np.save(landcover_path, mask)


def download_images(meta_load_path,
                    image_dir,
                    meta_save_path,
                    download_ls7=False,
                    cloud_fraction=CLOUD_FRACTION,
                    scene_limit=SCENE_LIMIT,
                    start_idx=0,
                    end_idx=None,
                    only_closest=False,
                    download_ir=False,
                    years_following=4,
                    center_tiles=True,
                    download_landcover=False,
                    download_sentinel=False):

    total_cts = []
    assert(os.path.exists(meta_load_path))
    metadata = pd.read_csv(meta_load_path,
                           header=0)
    
    print("Number total images: ", len(metadata))
    os.makedirs(image_dir, exist_ok=True)

    td = TileDownloader(cloud_fraction=cloud_fraction,
                        scene_limit=scene_limit,
                        only_closest=only_closest,
                        download_ir=download_ir,
                        years_following=years_following,
                        center_tiles=center_tiles,
                        download_sentinel=download_sentinel)

    metadict = defaultdict(list)

    if download_landcover:
        meta_headers.append(LANDCOVER_MAP_PATH_HEADER)
    
    num_not_found = 0
    idxs_captured = []
    for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):

        if index < start_idx:
            continue
        if end_idx is not None and index >= end_idx: 
            break

        lat, lon = row[LATITUDE_HEADER], row[LONGITUDE_HEADER]
        year = int(row[YEAR_HEADER])

        images_path = os.path.join(image_dir, str(index).zfill(4) + f"_yr-{year}_latlon-{round(lat, 2)}_{round(lon, 2)}")

        label = row[LABEL_HEADER]

        if download_landcover:
            landcover_path = os.path.join(images_path, "landcover.npy")
            if not os.path.exists(landcover_path):
                try:
                    td.download_landcover(lat, lon, landcover_path)
                except:
                    landcover_path = 'None'

        meta_path = os.path.join(images_path, "meta.json")
        rgb_path = os.path.join(images_path, "rgb")

        try:
            download_metadata, year_cts = td.download_images(
                lat, lon, year, images_path, download_ls7
            )
            download_metadata[NUM_SC_HEADER] = '-'.join([str(ct) for ct in year_cts])
            total_cts.append(year_cts)
            with open(meta_path, 'w') as f:
                json.dump(download_metadata, f)
            idxs_captured.append(index)

            for meta_header, data in download_metadata.items():
                metadict[meta_header].append(data)

        except Exception as e:
            print(traceback.format_exc())
            num_not_found += 1
            
        if index % 50 == 0:
            print(f'Writing to {meta_save_path}')
            new_metadata = metadata.take(idxs_captured)
            for meta_header, data in metadict.items():
                new_metadata[meta_header] = data

            new_metadata.to_csv(meta_save_path, header=True, index=True)


    print(f'Did not find {num_not_found}/{metadata.shape[0]} images')


    print(f'Writing to {meta_save_path}')
    new_metadata = metadata.take(idxs_captured)
    for meta_header, data in metadict.items():
        new_metadata[meta_header] = data

    new_metadata.to_csv(meta_save_path, header=True, index=True)

    return np.array(total_cts), num_not_found

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", help="Split of data to download", \
        type=str, choices=('train', 'val', 'test'), default='train')
    args = parser.parse_args()

    split = args.split
    print(f'Downloading {split} split')
    
    postdownload_csv = POSTDOWN_METADATA[split]
    predownload_csv = PREDOWN_METADATA[split]
    
    total_cts, num_not_found = download_images(predownload_csv, 
        HANSEN_V5_DIR / f'{split}/', 
        postdownload_csv, 
        start_idx=0,                                       
        end_idx=None,                                  
        download_ls7=True)
    
    np.save(HANSEN_V5_DIR / f'scene_cts_v5_{split}.npy', total_cts)
    with open(DATA_BASE_DIR / f'log_v5_{split}.txt', 'w') as f:
        f.write(f'Did not find {num_not_found}/{len(total_cts)} images')

if __name__ == "__main__":
    main()
