import sys

sys.path.insert(0, '../')
from util.util import *
from util.constants import *
from tqdm import tqdm
import logging
import pandas as pd
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)
from datetime import datetime


def gid2regions(data, gid_path=GOODE_R_ID_PATH):
    
    gid_df = pd.read_csv(gid_path, keep_default_na=True) #NaNs present
    print(f'Length before dropping missing: {len(gid_df)}')
    gid_df = gid_df.dropna()
    print(f'Length after dropping missing: {len(gid_df)}')
    gid_df.columns = ['GoodeR_ID', REGION_HEADER]

    gid_df[REGION_HEADER] = gid_df[REGION_HEADER].astype(int)
    gid_df[REGION_HEADER] -= gid_df[REGION_HEADER].min()
    print(f'Unique region indices: {gid_df[REGION_HEADER].unique()}')
    
    data = data.merge(gid_df, on="GoodeR_ID")
    return data

def insert_cont_col(data):
    """
    Inserts continent col. 
    NOTE: Decommissioned from v5 onwards!
    """
    orig_cols = list(data.columns)
    print(f'Read in list of headers {orig_cols}')
    if (LATITUDE_HEADER not in orig_cols) or (LONGITUDE_HEADER not in orig_cols):
        raise ValueException(f'Latlon information missing in headers {orig_cols}')
    header_idx = max(orig_cols.index(LONGITUDE_HEADER), orig_cols.index(LATITUDE_HEADER)) + 1
    if CONTINENT_HEADER in orig_cols:
        print('Continent header already exists!')
        return
    conts = []
    for r, row in tqdm(data.iterrows(), total=len(data)):
        conts.append(latlon_to_continent((row[LATITUDE_HEADER], row[LONGITUDE_HEADER])))

    assert len(conts) == len(data)    
    data.insert(header_idx, CONTINENT_HEADER, conts, allow_duplicates=False)
    return data

def translate_labels(data, base_idx=0):
    """
    Makes sure labels zero-indexed.
    NOTE: ALL labels must be represented AT LEAST ONCE in the dataset
    """
    offset = int(base_idx - data[LABEL_HEADER].min())
    if offset != 0:
        data[LABEL_HEADER] = data[LABEL_HEADER] + offset
    return data

def _col_is_anom(series):
    return (series['min'] < -1e20) or (abs(series['max']) > 1e20) 

def _anomalous_cols(df, min_threshold):
    anom = set()
    df_desc = df.describe()
    for col in df_desc.columns:
        if _col_is_anom(df_desc[col]):
            anom.add(col)
    return anom

def impute_missing_aux(data, min_threshold=-1e30,
                       impute_val=None):
    """
    Impute missin AUX variables
    impute_val can either be None, the train_mean (impute with mean) or 0 (impute with zero, as done in Curtis et al)
    """
    anom_cols = _anomalous_cols(data, min_threshold)
    for feat in AUX_FEATURE_HEADER:
        data.loc[data[feat] < min_threshold, feat] = np.nan
    if impute_val is None:
        data[AUX_FEATURE_HEADER] = data[AUX_FEATURE_HEADER].fillna(data[AUX_FEATURE_HEADER].mean(), inplace=False)
    else:
        data[AUX_FEATURE_HEADER] = data[AUX_FEATURE_HEADER].fillna(impute_val, inplace=False)
    return data, anom_cols

def standardize_aux(data, mean=None, std=None):
    if (mean is None) or (std is None):     
        mean = data[AUX_FEATURE_HEADER].mean()
        std = data[AUX_FEATURE_HEADER].std()
        data[AUX_FEATURE_HEADER] = (data[AUX_FEATURE_HEADER] - mean) / std
        return data, pd.DataFrame([mean, std])
    else:  
        data[AUX_FEATURE_HEADER] = (data[AUX_FEATURE_HEADER] - mean) / std
        return data
    
if __name__ == '__main__':
    norm_stats_path = os.path.join(HANSEN_V5_DIR, f'norm_stats_train.csv')
    logpath = os.path.join(HANSEN_V5_DIR, f'preprocessing_v5.log')
    for split in [HANSEN_TRAIN_V5_PATH_SHORT, HANSEN_VAL_V5_PATH_SHORT, HANSEN_TEST_V5_PATH_SHORT]: 
        loadpath = os.path.join(HANSEN_V5_DIR, f'postdownload_meta_{split}')
        savepath = os.path.join(HANSEN_V5_DIR, split)
            
        logging.basicConfig(filename= logpath, 
                            filemode='w', level=logging.DEBUG)
        
        ## snippet below prints to stderr while logging
        stderrLogger=logging.StreamHandler()
        stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        logging.getLogger().addHandler(stderrLogger)

        logging.info(f'Processing split {split}')
        os.makedirs(os.path.dirname(savepath), exist_ok=True)

        data = pd.read_csv(loadpath, header=0, index_col=0, na_filter=True)
        prev = data.columns
        
        data = translate_labels(data)
        data = gid2regions(data, gid_path=GOODE_R_ID_PATH)
        data, anoms = impute_missing_aux(data, impute_val=0)
        
        if split == HANSEN_TRAIN_V5_PATH_SHORT:
            
            data, train_stats = standardize_aux(data)
            train_stats.to_csv(norm_stats_path, header=True, index=True)
            logging.info(f'Saved training set stats to {norm_stats_path}')
            
        else:
            
            # load saved stats from train set
            train_stats = pd.read_csv(norm_stats_path, usecols=AUX_FEATURE_HEADER)
            logging.info(f'Loaded training set stats from {norm_stats_path}')
            train_mean = train_stats.iloc[0, :] 
            train_std = train_stats.iloc[1, :]  
            
            data = standardize_aux(data, train_mean, train_std)
        
        logging.info(f'Fixed {len(anoms)} columns w/ missing values')
        logging.info(f'Null values count:\n {data.isnull().sum()}')
        logging.info(f'New columns added: {[col for col in data.columns if col not in prev]}')
        
        data.to_csv(savepath, header=True, index=True)
        logging.info(f'Description stats:\n {data.describe()}')  
        logging.info(f'Saved new csv to path {savepath}')    
        logging.info(f'Done with {split}!')
        logging.info('=======')