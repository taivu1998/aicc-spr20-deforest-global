import sys
sys.path.insert(0, '../')

import json
import os
from os.path import join
import csv
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import re
import geopy
import pycountry_convert
from geopy.geocoders import Nominatim
import torch
import fire
from util.constants import *


LIGHTNING_CKPT_PATH = 'lightning_logs/version_0/checkpoints/'
LIGHTNING_TB_PATH = 'lightning_logs/version_0/'
LIGHTNING_METRICS_PATH = 'lightning_logs/version_0/metrics.csv'


class Args(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(args[0])

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            AttributeError("No such attribute: " + name)


def init_exp_folder(args):
    save_dir = os.path.abspath(args.get("save_dir"))
    exp_name = args.get("exp_name")
    exp_path = join(save_dir, exp_name)
    exp_metrics_path = join(exp_path, "metrics.csv")
    exp_tb_path = join(exp_path, "tb")
    global_tb_path = args.get("tb_path")
    global_tb_exp_path = join(global_tb_path, exp_name)

    # init exp path
    if os.path.exists(exp_path):
        raise FileExistsError(f"Experiment path [{exp_path}] already exists!")
    os.makedirs(exp_path, exist_ok=True)

    os.makedirs(global_tb_path, exist_ok=True)
    if os.path.exists(global_tb_exp_path):
        raise FileExistsError(f"Experiment exists in the global "
                              f"Tensorboard path [{global_tb_path}]!")
    os.makedirs(global_tb_path, exist_ok=True)

    # dump hyper-parameters/arguments
    json.dump(locals(),
              open(join(save_dir, exp_name, "args.json"), "w+"))

    # ln -s for metrics
    os.symlink(join(exp_path, LIGHTNING_METRICS_PATH),
               exp_metrics_path)

    # ln -s for tb
    os.symlink(join(exp_path, LIGHTNING_TB_PATH), exp_tb_path)
    os.symlink(exp_tb_path, global_tb_exp_path)
    

def display_event(image_dir, title="", version="", to_ignore=['cloud'], save_path=None):
    """
    Display utility - displays multiple jpg or png images in a directory side-by-side 
    
    Args: 
        image_dir(str) : Directory to read from
        title (str)    : Title for plot
    """
    all_ims = [path for path in glob.glob(image_dir+"*") if 'png' in path or 'jpg' in path]
    ims = []
    for path in all_ims:
        include = True
        for ignore_str in to_ignore:
            if ignore_str in path:
                include = False
        if include:
            ims.append(path)
    ims = sorted(ims)

    w=10
    h=10
    fig=plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
        
    columns = 5
    rows = 1
    raw_img_names = [(img.split('/')[-1]).split('.')[0] for img in ims]
    first_year = int(raw_img_names[0][:4])
    
    img_names = ["blank"] * columns
    img_list = [np.zeros((INPUT_ORIGINAL_HEIGHT, INPUT_ORIGINAL_WIDTH, NUM_RGB_CHANNELS))] * columns
    for i, img in enumerate(ims):
        if i < len(ims) - 1:
            year = int(raw_img_names[i][:4])
            img_names[year - first_year] = str(year)
            img_list[year - first_year] = np.asarray(Image.open(img, 'r'))
        else:
            img_names[-1] = "composite"
            img_list[-1] = np.asarray(Image.open(img, 'r'))
    if version != "":
        img_names = [img_name + "_" + version for img_name in img_names]
        
    for i in range(columns):
        frame = fig.add_subplot(rows, columns, i+1)
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        plt.imshow(img_list[i])
        plt.title(img_names[i])
    plt.show()

    if save_path is not None:
        plt.save_fig(save_path)


def compare_images(image_v3_path, image_v5_path):
    print("V3 Images:")
    display_event(image_v3_path, title="", version='v3', to_ignore=['cloud'], save_path=None)
    print("V5 Images:")
    display_event(image_v5_path, title="", version='v5', to_ignore=['cloud'], save_path=None)


def get_tuning_metrics(models_dir=SANDBOX_DIR, 
                       tracking_metric='avg_val_acc',
                       metrics=['avg_val_acc', 'avg_val_loss', 'avg_train_acc'],
                       metrics_file='metrics.csv', 
                       identifier='', 
                       prefix='',                   
                       save_path='hp_metrics.csv'):
    """
    Print accuracy and loss metrics from all models in a model directory.
    Particularly useful for obtaining metrics following a HP tuning run
    Each model folder MUST have a metrics_file csv file within 

    Args:
        models_dir (str)    : path to the directory of model dolders 
        identifier (str)    : target models must contain this string
        prefix     (str)    : target models must start with this string.
                               Useful when loading results from specific expts
    """
    
    data = []
    regex = re.compile(f'^{prefix}.*{identifier}')
    files = [path for path in os.scandir(models_dir) if regex.match(path.name)]
    print(f'Found {len(files)} files with prefix: {prefix} and identifier: {identifier}')
    for file in files:
        row = []
        row.append(file.name)
        csv_path = os.path.join(file, metrics_file)
        print(f'Processing {csv_path}')

        try:
            df = pd.read_csv(csv_path)
        except (pd.io.common.EmptyDataError, FileNotFoundError) as e:
            print(f'{e}. Skipping...')
            continue
        print(f'Found headers: {df.columns}')
        
        try:
            target_row = df[tracking_metric].idxmax(axis=0)
        except KeyError:
            print(f'{tracking_metric} not in columns for {file.name}. Skipping...')
            continue
        for m in metrics:
            if 'train' in m:
                ## Val indices lead train indices by 1 row
                row.append(df.loc[target_row - 1, m])
            else:
                row.append(df.loc[target_row, m])

        data.append(row)

    data_df = pd.DataFrame(data, columns=['Model']+metrics)
    data_df = data_df.sort_values(tracking_metric,
                                  ascending=False, axis=0)
    if save_path is not None:
        save_path = f'pre-{prefix}_id-{identifier}_' + save_path 
        data_df.to_csv(save_path)
    return data_df
    
def alpha2_to_continent(alpha2):
    """
    Converts 'country-alpha2 codes' to 'continent codes'
    Country-alpha2 are 2-letter country codes unique to each country
        Ref: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
        
    Continent codes are 2-char strings that denote each continent. 
        Ref: https://doc.bccnsoft.com/docs/php-docs-7-en/function.geoip-continent-code-by-name.html
        
    Args:
        alpha2 (str): country-alpha2 code. 2 characters, case-insensitive
        Ref: https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2
    Returns:
        Continent code (str): 2-char string, upper case 
    """
    return pycountry_convert.country_alpha2_to_continent_code(alpha2.upper())

    
def latlon_to_continent(latlon, nominatim_tout=30):
    """
    Converts latitude, longitude coordinates to 'continent codes'
        
    Continent codes are 2-char strings that denote each continent. 
        Ref: https://doc.bccnsoft.com/docs/php-docs-7-en/function.geoip-continent-code-by-name.html
        
    Args:
        latlon (tuple or list): latitude, longitude coordinates (in floats) of a location. Order-sensitive
    Returns:
        Continent code (str): 2-char string, upper case 
    """    
    geolocator = Nominatim(timeout=nominatim_tout)
    coords = ','.join([str(num) for num in latlon])
    location = geolocator.reverse(coords)
    country_code = location.raw['address']['country_code'].upper()
    if country_code in COUNTRY_EXCEPTIONS:
        return COUNTRY_EXCEPTIONS[country_code]
    else:
       return alpha2_to_continent(country_code)

def get_num_channels(model_args):
    num_channels = NUM_RGB_CHANNELS
    if not model_args['composite'] and not model_args['lrcn']:
        if model_args["first_last"]:
            num_channels *= 2
        else:
            num_channels *= MAX_IMGS_PER_LOCATION
    return num_channels

def copy_weights(model, conv_weight, num_channels, model_name):
        assert(num_channels % 3 == 0), "Number of channels not divisible by 3"
        state_dict_copy = model.state_dict()
        conv_weight = torch.cat([conv_weight] + [conv_weight[:, :, :, :]] *
                                int((num_channels - 3) / 3), dim=1)
        if 'densenet' in model_name:
            state_dict_copy["features.conv0.weight"] = conv_weight
        elif 'resnet' in model_name:
            state_dict_copy["conv1.weight"] = conv_weight
        else:
            raise RuntimeError('Only DenseNet and ResNet implemented')
        model.load_state_dict(state_dict_copy)
    
if __name__ == "__main__":
    fire.Fire()
