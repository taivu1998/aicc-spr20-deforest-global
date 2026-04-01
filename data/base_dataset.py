import numpy as np
import torch
import pickle
import glob
import pandas
import sys
import os
import random
from PIL import Image, ImageEnhance
from torchvision import transforms
from util.constants import *  
from pathlib import *

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 image_path,
                 data_split,
                 transforms,
                 regions,
                 img_option,
                 first_last,
                 lrcn,
                 load_polygon_loss,
                 late_fusion_regions,
                 load_aux,
                 load_mode, 
                 year_cutoff,
                 deterministic=True,
                 padding='end',
                 num_images=1
                ):
        """
        Args:
            image_path:        path to image dir
            data_split:        one of ['train', 'val', 'test']
            transforms:        augmentation transform
            img_option:        options for loading images
            num_images:        Number of images to load per event. 
            regions:           List of continent headers if region-specific models trained, else None (global model)
            first_last:        only returns oldest and most recent images for an event
            lrcn:              sequence of images rather than one image
            deterministic:     <Unsure>
            load_polygon_loss: return the area of loss polygon as well as image
            late_fusion_regions: one of ['none', 'latlon', 'onehot']
            padding:           one of {'start', 'end'}
            load_mode:         one of {'scene', 'annual', 'random'} each image loaded represents a scene/year depending on granularity.\
            if 'random, load each time element as scene or annual composite with equal probability. This is IGNORED if composite is True. 
            year_cutoff:       year of cutoff when loading images FOR TRAINING. 
        """

        self._image_path = image_path
        self._data_split = data_split
        self._image_info = None
        self._img_option = img_option
        self._first_last = first_last
        self._lrcn = lrcn
        self._transforms = transforms
        self._deterministic = deterministic
        self._load_polygon_loss = load_polygon_loss
        self._load_aux = load_aux
        self._regions = regions
        self._late_fusion_regions = late_fusion_regions
        self._padding = padding
        self._load_mode = load_mode
        self._year_cutoff = year_cutoff
        self.process_file()
        
    def process_file(self):
        raise Exception(NotImplementedError)

    def __len__(self):
        return self._image_info.shape[0]
    
    def _get_label(self, index):
        # check if index in bounds?
        label = self._image_info.iloc[index][LABEL_HEADER]
        label = torch.tensor(label, dtype=torch.long)
        return label
    
    def _load_img(self, path):
        pil_image = Image.open(path).convert('RGB')
        pil_image = ImageEnhance.Brightness(pil_image).enhance(1.5)
        image = np.array(pil_image)
        return image

    def _get_composite_path(self, index):
        im_dir = Path(self._image_info.iloc[index][IMG_PATH_HEADER])
        return im_dir / self._image_info.iloc[index][IMG_OPTION_COMPOSITE]
        
    def _get_image(self, index):
        image_list = list()
        im_dir = Path(self._image_info.iloc[index][IMG_PATH_HEADER])
        if not os.path.isdir(im_dir):
            raise FileNotFoundError(f"Directory {im_dir} does not exist!")

        if not self._lrcn:
            if self._img_option == IMG_OPTION_COMPOSITE:
                im_path = self._get_composite_path(index)
            elif self._img_option == IMG_OPTION_RANDOM:
                rgb_dir = im_dir / 'rgb'
                if self._load_mode == 'scene':
                    glob_results = sorted(Path(rgb_dir).glob('2*.png'))
                    glob_results = [glob_result for glob_result in glob_results if '_annual' not in str(glob_result)]
                    if len(glob_results) == 0: 
                        glob_results = sorted(Path(rgb_dir).glob('2*.png'))
                elif self._load_mode == 'annual':
                    glob_results = sorted(Path(rgb_dir).glob('*_annual.png'))
                elif self._load_mode == 'annualorscene':
                    glob_results = sorted(Path(rgb_dir).glob('2*.png'))
                elif self._load_mode == 'all':
                    glob_results = sorted(Path(rgb_dir).glob('2*.png'))
                    glob_results.append(self._get_composite_path(index))
                if len(glob_results) == 0:
                    glob_results = [self._get_composite_path(index)]
                im_path = random.choice(glob_results)
            else:
                rgb_dir = im_dir / 'rgb'
                glob_results = sorted(list(Path(rgb_dir).glob('2*.png')) + list(Path(rgb_dir).glob('*_annual.png')))
                if self._img_option == IMG_OPTION_CLOSEST_YEAR:
                    im_path = glob_results[0]
                elif self._img_option == IMG_OPTION_FURTHEST_YEAR:
                    im_path = glob_results[len(glob_results)-1]
                else:
                    raise ValueError("Invalid value for img_option")
            if not os.path.exists(im_path):
                raise Exception(f"{im_path} does not exist")
            image_list.append(self._load_img(im_path))
            
        if self._lrcn:
            rgb_dir = im_dir / 'rgb'            
            if self._load_mode == 'scene':
                glob_results = sorted(Path(rgb_dir).glob('2*.png'))
            elif self._load_mode == 'annual':
                glob_results = sorted(Path(rgb_dir).glob('*_annual.png'))
            else:
                raise NotImplementedError("Random load mode not implemented!")

            ## Set what images we want to add
            if self._first_last:
                glob_results = [glob_results[0], glob_results[-1]]
            
            num_images = len(glob_results)
            
            if num_images == 0:
                # load composite as backup
                im_path = self._get_composite_path(index)
                glob_results = [im_path]
                       
            ## Set ordering of images
            for file in glob_results:
                image_list.append(self._load_img(file))
                
            if not self._first_last:
                # Lower accuracy when replacing missing images with blank images rather than stacking at the end
                loaded_images = len(image_list)
                blank_shape = image_list[0].shape # image list cannot be empty
                blanks = [
                    np.zeros(blank_shape, dtype=np.float32)
                    for _ in range(loaded_images, MAX_IMGS_PER_LOCATION)
                ]
                if self._padding == 'end':
                    image_list = image_list + blanks
                elif self._padding == 'start':
                    image_list = blanks + image_list
                else:
                    raise NotImplementedError("Padding other than start and end not implemented!")
        return image_list

    def _get_loss_areas(self, index):
        if self._first_last:
            raise RuntimeError('Polygon model not supported for first_last.')
        elif self._img_option == IMG_OPTION_COMPOSITE:
            num_years = 1
        else:
            num_years = MAX_IMGS_PER_LOCATION

        lat = self._image_info.iloc[index][LATITUDE_HEADER]
        lon = self._image_info.iloc[index][LONGITUDE_HEADER]
        loss_year = self._image_info.iloc[index][YEAR_HEADER]
        loss_area_in_loss_year = 0
        polygon_dir = POLYGON_DIRS[self._data_split]
        shapefile_path = polygon_dir / f'{round(lat, 5)}_{round(lon, 5)}'

        if loss_year >= BASE_YEAR and loss_year <= END_YEAR:
            shapefile_path_year = shapefile_path / f'{loss_year}'
            if os.path.isdir(shapefile_path_year):
                for shape_file in glob.glob(str(shapefile_path_year) + '/*'):
                    with open(shape_file, 'rb') as f:
                        polygon = pickle.load(f)
                        loss_area_in_loss_year += polygon.area
            loss_area_in_loss_year /= MAX_LOSS_AREA
        
        return loss_area_in_loss_year

    def class_weights(self):
        freq = self._image_info[LABEL_HEADER].value_counts(
            normalize=True).sort_index()
        return 1.0 / freq

    def _get_latlon(self, index):
        lat = float(self._image_info.iloc[index][LATITUDE_HEADER])
        lon = float(self._image_info.iloc[index][LONGITUDE_HEADER])
        return lat, lon

    def _get_region(self, index):
        return int(self._image_info.iloc[index][REGION_HEADER])
    
    def _get_aux_features(self, index):
        aux_features = self._image_info.iloc[index][AUX_FEATURE_HEADER].to_dict()
        return aux_features
