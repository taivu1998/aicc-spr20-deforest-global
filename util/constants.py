f"""Define constants to be used throughout the repository."""
from pathlib import Path
import torchvision.transforms as T
from collections import OrderedDict
import imgaug.augmenters as iaa
import imgaug as ia
import numpy as np

# Dataset constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

## Continent codes
CONTINENTS = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA'] #Legacy version 

REGIONS = ['na', 'la', 'eu', 'af', 'as', 'sea', 'oc'] #Updated and same order as Nature paper

REGION_EMBEDDINGS = np.eye(len(REGIONS))


## Countries that don't have designated continents per the package being used
COUNTRY_EXCEPTIONS = {'TF':'OC', # TF is in the South, island of 200-400 scientists
                      'VA': 'EU', # Vatican is in EU
                      'PN': 'OC', # PN are group of islands in the Southern Pac Ocean
                      'SX': 'NA', # SX is off the coast of Mexico, north of the equator
                      'TL': 'OC', # Island in Indian Ocean, closer to Aus than Asia 
                      'UM': 'NA', 
                      'EH': 'AF',
                      'AQ': 'NA' # Let's assume Antarctica belongs to North America by default. After all so does Canada
                        }

# US latitude/longitude boundaries
US_N = 49.4
US_S = 24.5
US_E = -66.93
US_W = -124.784

#Define time
PRESENT_YEAR = 2020 # rip
CUTOFF_YEAR = 2012 #ignore events before this year
BASE_YEAR = 2000
END_YEAR = 2018

# Main paths
SHARED_DEEP_DIR = Path('/deep/group/aicc-bootcamp/deforest-global/')
DATA_BASE_DIR = SHARED_DEEP_DIR / 'data'
MODEL_BASE_DIR = SHARED_DEEP_DIR / 'models'
PREDOWNLOAD_DIR = DATA_BASE_DIR / 'pre_download'
SANDBOX_DIR = MODEL_BASE_DIR / 'sandbox'
TB_DIR = SANDBOX_DIR / 'tb'

# Dataset constants
HANSEN_DATASET_NAME = 'hansen'
DATASET_NAMES = [HANSEN_DATASET_NAME]
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'
DATA_SPLITS = [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT]

# Hansen constants
HANSEN_DIR = DATA_BASE_DIR
HANSEN_V5_DIR = HANSEN_DIR / 'curtis_v5'
HANSEN_TRAIN_PATH = HANSEN_DIR / 'train_latlon_v1.csv'

HANSEN_TRAIN_V2_PATH = HANSEN_DIR / 'train_latlon_v2.csv'
HANSEN_VAL_V2_PATH = HANSEN_DIR / 'val_latlon_v2.csv'

HANSEN_TRAIN_V3_PATH_SHORT = 'train_v3.csv'
HANSEN_VAL_V3_PATH_SHORT = 'val_v3.csv'
HANSEN_TEST_V3_PATH_SHORT = 'test_v3.csv'

HANSEN_TRAIN_V3_PATH = HANSEN_DIR / HANSEN_TRAIN_V3_PATH_SHORT
HANSEN_VAL_V3_PATH = HANSEN_DIR / HANSEN_VAL_V3_PATH_SHORT
HANSEN_TEST_V3_PATH = HANSEN_DIR / HANSEN_TEST_V3_PATH_SHORT

HANSEN_TRAIN_CURTIS_PATH_SHORT = 'train_curtis.csv'
HANSEN_VAL_CURTIS_PATH_SHORT = 'val_curtis.csv'
HANSEN_TEST_CURTIS_PATH_SHORT = 'test_curtis.csv'

HANSEN_TRAIN_CURTIS_PATH = HANSEN_DIR / HANSEN_TRAIN_CURTIS_PATH_SHORT
HANSEN_VAL_CURTIS_PATH = HANSEN_DIR / HANSEN_VAL_CURTIS_PATH_SHORT
HANSEN_TEST_CURTIS_PATH = HANSEN_DIR / HANSEN_TEST_CURTIS_PATH_SHORT

## paths for v5 data
TRAIN_V5_POSTDOWN_PATH = HANSEN_V5_DIR / 'postdownload_meta_train_v5.csv'
VAL_V5_POSTDOWN_PATH = HANSEN_V5_DIR / 'postdownload_meta_val_v5.csv'
TEST_V5_POSTDOWN_PATH = HANSEN_V5_DIR / 'postdownload_meta_test_v5.csv'

HANSEN_TRAIN_V5_PATH_SHORT = 'train_v5.csv'
HANSEN_VAL_V5_PATH_SHORT = 'val_v5.csv'
HANSEN_TEST_V5_PATH_SHORT = 'test_v5.csv'

GOODE_R_ID_PATH = Path('/deep/group/aicc-bootcamp/deforest-global/data/curtis/GoodeR_Boundaries_Region.csv')

TRAIN_V5_PREDOWN = DATA_BASE_DIR / 'curtis_processed/v5_training_all.csv'
VAL_V5_PREDOWN = DATA_BASE_DIR / 'curtis_processed/v5_validation_all.csv'
TEST_V5_PREDOWN = DATA_BASE_DIR / 'curtis_processed/v5_test_all.csv'

HANSEN_TRAIN_V5_PATH = HANSEN_V5_DIR / HANSEN_TRAIN_V5_PATH_SHORT
HANSEN_VAL_V5_PATH = HANSEN_V5_DIR / HANSEN_VAL_V5_PATH_SHORT
HANSEN_TEST_V5_PATH = HANSEN_V5_DIR / HANSEN_TEST_V5_PATH_SHORT

NLCD_PATH = DATA_BASE_DIR / 'landcover'
NLCD_LABELS_PATH = NLCD_PATH / 'labels'

PREDOWN_METADATA = {
    TRAIN_SPLIT : TRAIN_V5_PREDOWN,
    VAL_SPLIT : VAL_V5_PREDOWN,
    TEST_SPLIT : TEST_V5_PREDOWN,
}

POSTDOWN_METADATA = {
    TRAIN_SPLIT : TRAIN_V5_POSTDOWN_PATH,
    VAL_SPLIT : VAL_V5_POSTDOWN_PATH,
    TEST_SPLIT : TEST_V5_POSTDOWN_PATH
}


METADATA_FILES = {
    TRAIN_SPLIT : '/deep/group/aicc-bootcamp/deforest-global/data/curtis_processed/v5_training_all.csv',
    VAL_SPLIT : '/deep/group/aicc-bootcamp/deforest-global/data/curtis_processed/v5_validation_all.csv',
    TEST_SPLIT : '/deep/group/aicc-bootcamp/deforest-global/data/curtis_processed/v5_test_all.csv'
}

POLYGON_DIR = DATA_BASE_DIR / 'polygon_v5'
POLYGON_TRAIN_DIR = POLYGON_DIR / 'train'
POLYGON_VAL_DIR = POLYGON_DIR / 'val'
POLYGON_TEST_DIR = POLYGON_DIR / 'test'

POLYGON_DIRS = {
    TRAIN_SPLIT : POLYGON_TRAIN_DIR,
    VAL_SPLIT : POLYGON_VAL_DIR,
    TEST_SPLIT : POLYGON_TEST_DIR,
}

SECO_PRETRAINED_PATHS = [
    '/deep/group/aicc-bootcamp/deforest-global/pretrained/seco_resnet18_100k_encoder.ckpt',
    '/deep/group/aicc-bootcamp/deforest-global/pretrained/seco_resnet18_1m_encoder.ckpt',
    '/deep/group/aicc-bootcamp/deforest-global/pretrained/seco_resnet50_100k_encoder.ckpt',
    '/deep/group/aicc-bootcamp/deforest-global/pretrained/seco_resnet50_1m_encoder.ckpt'
]

# Label constants
HANSEN_LABELS = ['Commodity Driven Deforestation', 'Shifting Agriculture',
                 'Forestry', 'Wildfire', 'Urbanization',
                 'Other Natural Disturbance', 'Uncertain']
HANSEN_LABELS_V3 = ['Commodity Driven Deforestation', 'Shifting Agriculture',
                    'Forestry', 'Wildfire', 'Urbanization']
HANSEN_IGNORED_LABELS = ['Other Natural Disturbance', 'Uncertain']
HANSEN_IGNORED_LABEL_IDXS = [HANSEN_LABELS.index(label)
                             for label in HANSEN_IGNORED_LABELS]

HANSEN_NUM_CLASSES = len(HANSEN_LABELS_V3)

TEMPORAL_RANGE = 4 #Max number of years of data to stack


# NOTE: based on successfully downloaded images from
# train_latlon_v3.csv. Should be adjusted when missing images found.
HANSEN_V3_TRAIN_FREQS = [848, 1166, 1549, 749, 222]
HANSEN_V3_NUM_EXS = sum(HANSEN_V3_TRAIN_FREQS)
HANSEN_V3_CLASS_WEIGHTS = [HANSEN_V3_NUM_EXS / freq
                           for freq in HANSEN_V3_TRAIN_FREQS]

POLYGON_SHAPE = 'Polygon'
MULTIPOLYGON_SHAPE = 'MultiPolygon'


#HANSEN_TRAIN_SPLIT, HANSEN_VAL_SPLIT = .85, .15

DATASET_LABELS_NAMES = {
    HANSEN_DATASET_NAME: {
        'label_names': HANSEN_LABELS_V3,
        'labels': list(range(len(HANSEN_LABELS_V3))),
    }
}

# NOTE: based on successfully downloaded images from
# train_latlon_year.csv. Should be adjusted when missing images found.

# Dataset constants
RESIZE_HEIGHT = 300  # TODO: These dims should be cross-validated
RESIZE_WIDTH = 300
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
SMALL_RESIZE_HEIGHT = 150
SMALL_RESIZE_WIDTH = 150
SMALL_INPUT_HEIGHT = 120
SMALL_INPUT_WIDTH = 120
INPUT_ORIGINAL_HEIGHT = 666
INPUT_ORIGINAL_WIDTH = 666
SHAPE_SIZE_THRESHOLD = 2
FORESTED_THRESHOLD = 30

NUM_RGB_CHANNELS = 3
NUM_IR_CHANNELS = 3
NUM_MASKED_CHANNELS = 1

MAX_IMGS_PER_LOCATION = 4
MAX_LOSS_AREA = 331836.0

# Both PIL and imgaug complain about having negative values, 
# and we use -100 as a target sentinel to ignore loss for 
# certain outputs. So, we use LABEL_IGNORE_VALUE 
# and switch these to -100 (LOSS_IGNORE_VALUE) after the transforms.
LABEL_IGNORE_VALUE = 255
LOSS_IGNORE_VALUE = -100

NO_AUGMENT_TRANSFORMS = [iaa.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
                         T.ToTensor()]

LABEL_IGNORE_VALUE = 255
LOSS_IGNORE_VALUE = -100

AUGMENTATION_TRANSFORM = {
    "none": iaa.Identity(),
    "flip": iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)]),
    "affine":
    iaa.SomeOf(2, [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Rot90(1, 3)
    ]),
    "cloud":
    iaa.Sequential([
        iaa.SomeOf(2, [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Rot90(1, 3)]),
        iaa.Sometimes(0.5,
                      iaa.OneOf([
                          iaa.Clouds(),
                          iaa.Fog(),
                          iaa.Snowflakes()]))]),
    "sap":
    iaa.SomeOf(2, [
        iaa.SaltAndPepper([0.0, 0.01]),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            rotate=(-5, 5),
            cval=LABEL_IGNORE_VALUE,
            mode='constant')
    ]),
    "random": iaa.RandAugment(n=2, m=9),
    "aggressive":
    iaa.SomeOf(2, [
        iaa.SaltAndPepper([0.0, 0.1]),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            rotate=(-5, 5),
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.Affine(
            shear=(-5, 5),
            cval=LABEL_IGNORE_VALUE,
            mode='constant'),
        iaa.ElasticTransformation(alpha=(0.0, 40.0), sigma=(4.0, 8.0))
    ])
}


RESIZE_CROP_TRANSFORM = {False: {
    TRAIN_SPLIT: iaa.Sequential([
        iaa.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        iaa.CropToFixedSize(INPUT_WIDTH, INPUT_HEIGHT)]),
    VAL_SPLIT: iaa.Sequential([
        iaa.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        iaa.CenterCropToFixedSize(INPUT_WIDTH, INPUT_HEIGHT)]),
    TEST_SPLIT: iaa.Sequential([
        iaa.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        iaa.CenterCropToFixedSize(INPUT_WIDTH, INPUT_HEIGHT)])},
    True: {
    TRAIN_SPLIT: iaa.Sequential([
        iaa.Resize((SMALL_RESIZE_WIDTH, SMALL_RESIZE_HEIGHT)),
        iaa.CropToFixedSize(SMALL_INPUT_WIDTH, SMALL_INPUT_HEIGHT)]),
    VAL_SPLIT: iaa.Sequential([
        iaa.Resize((SMALL_RESIZE_WIDTH, SMALL_RESIZE_HEIGHT)),
        iaa.CropToFixedSize(SMALL_INPUT_WIDTH, SMALL_INPUT_HEIGHT)]),
    TEST_SPLIT: iaa.Sequential([
        iaa.Resize((SMALL_RESIZE_WIDTH, SMALL_RESIZE_HEIGHT)),
        iaa.CropToFixedSize(SMALL_INPUT_WIDTH, SMALL_INPUT_HEIGHT)])}}

TOTENSOR_TRANSFORM = T.ToTensor()

IMAGE_NET_TRANSFORMS = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

# DL Band names
RED_BAND = 'red'
GREEN_BAND = 'green'
BLUE_BAND = 'blue'
NIR_BAND = 'nir'
SWIR1_BAND = 'swir1'
SWIR2_BAND = 'swir2'
CLOUD_MASK_BAND = 'cloud-mask'
CLOUD_MASK_BAND_LS7 = "derived:visual_cloud_mask"
NDVI_BAND = "ndvi"
NDVI_BAND_LS7 = "derived:ndvi"
BRIGHT_MASK_BAND = 'bright-mask'
CIRRUS_BAND = 'cirrus'
RGB_BANDS = [RED_BAND, GREEN_BAND, BLUE_BAND]
IR_BANDS = [NIR_BAND, SWIR1_BAND, SWIR2_BAND]
CLOUD_BANDS = [CLOUD_MASK_BAND, BRIGHT_MASK_BAND, CIRRUS_BAND]
BANDS_LS8 = RGB_BANDS + IR_BANDS + CLOUD_BANDS
BANDS_LS8_NO_IR = RGB_BANDS + CLOUD_BANDS
BANDS_LS7 = RGB_BANDS + IR_BANDS + [CLOUD_MASK_BAND_LS7] + [NDVI_BAND_LS7]
BANDS_LS7_NO_IR = RGB_BANDS + [CLOUD_MASK_BAND_LS7] + [NDVI_BAND_LS7]

BANDS = BANDS_LS8

TREECOVER_BAND = 'treecover2000'
LOSSYEAR_BAND = 'lossyear'

SCENE_LIMIT = 70
SINGLE_IMG_CLOUD_FRAC = 0.005
SMALL_COMP_SC_NUM = 5
SMALL_COMP_CLOUD_FRAC = 0.05

SINGLE_IMG_DOWNLOAD_METHOD = 'single image'
SMALL_COMPOSITE_DOWNLOAD_METHOD = 'small composite'
FULL_COMPOSITE_DOWNLOAD_METHOD = 'full composite'

# Tile download constants
KM_TO_DEG = 0.008
HANSEN_TILE_SIZE_KM = 10
HANSEN_TILE_SIZE_PX = 2
CLOUD_FRACTION = 0.5

# DL product constants
LANDSAT8_TIER1_PRODUCT_NAME = 'landsat:LC08:01:T1:TOAR'
LANDSAT8_PRE_COLLECTION_PRODUCT_NAME = 'landsat:LC08:PRE:TOAR'
LANDSAT7_PRE_COLLECTION_PRODUCT_NAME = 'landsat:LE07:PRE:TOAR'
NLCD_PRODUCT_NAME = 'nlcd:land_cover'
LANDSAT5_PRE_COLLECTION_PRODUCT_NAME = 'landsat:LT05:PRE:TOAR'
SENTINEL2_PRODUCT_NAME = 'sentinel-2:L1C'


LANDSAT8_TIER1_PRODUCT_RES = 15
LANDSAT7_PRE_COLLECTION_PRODUCT_RES = 15
NLCD_PRODUCT_RES = 15

GLOBAL_FOREST_CHANGE_PRODUCT_ID = '42b24cbb9a71ed9beb967dbad04ea61d7331d5af:global_forest_change_v0'
LAT_LON_EPSG = 'epsg:4326'

# Data CSV headers
LABEL_HEADER = 'label'
X_CENTROID_HEADER = 'x_coord'
Y_CENTROID_HEADER = 'y_coord'

LATITUDE_HEADER = 'latitude'
LONGITUDE_HEADER = 'longitude'
IMG_PATH_HEADER = 'image_paths'
YEAR_HEADER = 'year'
INDICES_HEADER = 'Event_index'
NUM_IMAGES_HEADER = 'event_images'
NUM_SC_HEADER = 'num_scenes'
IR_PATH_HEADER = 'ir_paths'
DOWNLOAD_METHOD_HEADER = 'download_method'
MULTI_IMAGE_HEADER = 'multi_yr_img_path'
MULTI_IR_HEADER = 'multi_yr_ir_path'

CONTINENT_HEADER = 'continent'
REGION_HEADER = 'region'

SHAPEFILE_HEADER = 'shape_paths'
AREA_HEADER = 'area_ha'

AUX_FEATURE_HEADER = [
    'Fire_Brightness_10kMax', 
    'Fire_Brightness_10kMax1kMean',
    'Fire_Brightness_10kMax1kSum',	
    'Fire_Brightness_10kMean',
    'Fire_Brightness_10kMean1kMax',
    'Fire_Brightness_10kMean1kSum',
    'Fire_Brightness_10kSum',
    'Fire_Count_10kMax',
    'Fire_Count_10kMax1kMean',
    'Fire_Count_10kMax1kSum',
    'Fire_Count_10kMean',
    'Fire_Count_10kMean1kMax',
    'Fire_Count_10kMean1kSum',
    'Fire_Count_10kSum',
    'Fire_FRP_10kMax',
    'Fire_FRP_10kMax1kMean',
    'Fire_FRP_10kMax1kSum',
    'Fire_FRP_10kMean',
    'Fire_FRP_10kMean1kMax',
    'Fire_FRP_10kMean1kSum',
    'Fire_FRP_10kSum',
    'Goode_FireLoss_10kMax',
    'Goode_FireLoss_10kMax1kMean',
    'Goode_FireLoss_10kMax1kSum',
    'Goode_FireLoss_10kMean',
    'Goode_FireLoss_10kMean1kMax',
    'Goode_FireLoss_10kMean1kSum',
    'Goode_FireLoss_10kSum',
    'Goode_Gain_10kMax',
    'Goode_Gain_10kMax1kMean',
    'Goode_Gain_10kMax1kSum',
    'Goode_Gain_10kMean',
    'Goode_Gain_10kMean1kMax',
    'Goode_Gain_10kMean1kSum',
    'Goode_Gain_10kSum',
    'Goode_LandCover_Deciduous_BroadLeaf_3',
    'Goode_LandCover_Evergreen_BroadLeaf_2',
    'Goode_LandCover_Mixed_Other_4',
    'Goode_LandCover_Needleleaf_1',
    'Goode_LossYearDiff_10kMax',
    'Goode_LossYearDiff_10kMax1kMean',
    'Goode_LossYearDiff_10kMax1kSum',
    'Goode_LossYearDiff_10kMean',
    'Goode_LossYearDiff_10kMean1kMax',
    'Goode_LossYearDiff_10kMean1kSum',
    'Goode_LossYearDiff_10kSum',
    'Goode_LossYearDiff_1k_10kMax',
    'Goode_LossYearDiff_1k_10kMean',
    'Goode_LossYearDiff_1k_10kSum',
    'Goode_Loss_10kMax',
    'Goode_Loss_10kMax1kMean',
    'Goode_Loss_10kMax1kSum',
    'Goode_Loss_10kMean',
    'Goode_Loss_10kMean1kMax',
    'Goode_Loss_10kMean1kSum',
    'Goode_Loss_10kSum',
    'Goode_NetMean',
    'Goode_Population2000_10kMax',
    'Goode_Population2000_10kMax1kMean',
    'Goode_Population2000_10kMax1kSum',
    'Goode_Population2000_10kMean',
    'Goode_Population2000_10kMean1kMax',
    'Goode_Population2000_10kMean1kSum',
    'Goode_Population2000_10kSum',
    'Goode_Population2015_10kMax',
    'Goode_Population2015_10kMax1kMean',
    'Goode_Population2015_10kMax1kSum',
    'Goode_Population2015_10kMean',
    'Goode_Population2015_10kMean1kMax',
    'Goode_Population2015_10kMean1kSum',
    'Goode_Population2015_10kSum',
    'Goode_PopulationDifference20002015_10kMax',
    'Goode_PopulationDifference20002015_10kMax1kMean',
    'Goode_PopulationDifference20002015_10kMax1kSum',
    'Goode_PopulationDifference20002015_10kMean',
    'Goode_PopulationDifference20002015_10kMean1kMax',
    'Goode_PopulationDifference20002015_10kMean1kSum',
    'Goode_PopulationDifference20002015_10kSum',
    'Tree_cover2000_10kMax',
    'Tree_cover2000_10kMax1kMean',
    'Tree_cover2000_10kMax1kSum',
    'Tree_cover2000_10kMean',
    'Tree_cover2000_10kMean1kMax',
    'Tree_cover2000_10kMean1kSum',
    'Tree_cover2000_10kSum'
]

AUX_SUBSET_FEATURE_HEADER = [
    'Goode_Population2015_10kMax1kSum',
    'Goode_Population2015_10kMax1kMean',
    'Goode_PopulationDifference20002015_10kMax1kMean',
    'Goode_PopulationDifference20002015_10kMax1kSum',
    'Goode_Population2000_10kMax',
    'Goode_Population2015_10kMax',
    'Goode_PopulationDifference20002015_10kMax',
    'Goode_Population2000_10kMax1kMean'
]

### HANSEN V4 specific ###
IMG_OPTION_CLOUD = 'least_cloudy'
IMG_OPTION_CLOSEST_YEAR = 'closest_year'
IMG_OPTION_FURTHEST_YEAR = 'furthest_year'
IMG_OPTION_COMPOSITE = 'composite'
IMG_OPTION_RANDOM = 'random'
NUM_IMGS_DOWNLOADED = 'num_imgs_downloaded'
IMG_COMPOSITE_IS_LS8 = 'composite_is_landsat8'


SCENE_LIMIT = 200
NDVI_IMG_MEAN_LS7 = 48000
SINGLE_IMG_CLOUD_FRAC = 0.005
SINGLE_IMG_CLOUD_FRAC_LS7 = 0.015
SMALL_COMP_SC_NUM = 5
SMALL_COMP_CLOUD_FRAC = 0.05

SINGLE_IMG_DOWNLOAD_METHOD = 'single image'
SMALL_COMPOSITE_DOWNLOAD_METHOD = 'small composite'
FULL_COMPOSITE_DOWNLOAD_METHOD = 'full composite'



###

## col names before download
HANSEN_v4_DOWNLOAD_META_COLNAMES = [LABEL_HEADER, 
                                 X_CENTROID_HEADER,
                                 Y_CENTROID_HEADER,
                                 LATITUDE_HEADER,
                                 LONGITUDE_HEADER,
                                 YEAR_HEADER
                                 ]

## col names immediately after download
HANSEN_v4_POSTDOWNLOAD_COLNAMES = HANSEN_v4_DOWNLOAD_META_COLNAMES + \
[NUM_IMGS_DOWNLOADED, 
IMG_OPTION_CLOUD,
IMG_OPTION_CLOSEST_YEAR,
IMG_OPTION_FURTHEST_YEAR,
IMG_OPTION_COMPOSITE,
IMG_COMPOSITE_IS_LS8,
IMG_PATH_HEADER,
NUM_SC_HEADER
]


###### V3 ######
HANSEN_DOWNLOAD_META_COLNAMES = [LABEL_HEADER, 
                                 LATITUDE_HEADER,
                                 LONGITUDE_HEADER,
                                 YEAR_HEADER
                                 ]

# Train/Valid data CSV headers
HANSEN_ORIGINAL_META_COLNAMES = [LABEL_HEADER,
                        LATITUDE_HEADER,
                        LONGITUDE_HEADER,
                        YEAR_HEADER,
                        IMG_PATH_HEADER,    
                        NUM_IMAGES_HEADER,
                        NUM_SC_HEADER,
                        DOWNLOAD_METHOD_HEADER, 
                        MULTI_IMAGE_HEADER,
                        MULTI_IR_HEADER]

HANSEN_META_COLNAMES = [LABEL_HEADER,
                        LATITUDE_HEADER,
                        LONGITUDE_HEADER,
                        CONTINENT_HEADER,
                        YEAR_HEADER,
                        IMG_PATH_HEADER,    
                        NUM_IMAGES_HEADER,
                        NUM_SC_HEADER,
                        DOWNLOAD_METHOD_HEADER, 
                        MULTI_IMAGE_HEADER,
                        MULTI_IR_HEADER]

# Model architecture
NUM_INPLANES = 64

SHAPE_META_COLNAMES = [LABEL_HEADER,
                      LATITUDE_HEADER,
                      LONGITUDE_HEADER,
                      YEAR_HEADER,
                      AREA_HEADER,
                      IMG_PATH_HEADER,
                      NUM_SC_HEADER]

# SEG_LEGEND_PATH = 'util/color_legend.png'

# CAM constants
MODEL2CAM_LAYER = {"DenseNet121": "model.features",
                   "ResNet152": "model.layer4.2.conv3",
                   "Inceptionv4": "model.features.21.branch3.1.conv",
                   "ResNet101": "model.layer4.2.conv3"}
CAM_DIR = 'cams'
CAM_PATH = 'CAM_path'
IMAGE_PATH = 'image_path'
TARGET_PATH = 'target_path'
NUMPY_PATH = 'numpy_path'
PROB = 'probability'
PRED = 'prediction'
INDEX = 'index'
TARGET = 'target'

# Geo encoding constants
LAT_MAX = 69.54039360899999
LAT_MIN = -46.8300900321
LON_MAX = 178.358085338
LON_MIN = -159.750944987
