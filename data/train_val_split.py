import sys
import fire
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append('util')
from constants import *

HANSEN_TRAIN_SPLIT, HANSEN_VAL_SPLIT = .85, .15

def split_train_val(metadata_file, train_metadata_file,
                    valid_metadata_file, val_size=HANSEN_VAL_SPLIT):
    metadata = pd.read_csv(metadata_file)
    y = metadata[LABEL_HEADER]
    X = metadata.drop(LABEL_HEADER, axis=1)
    
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X, y, test_size=val_size, random_state=0)
    data_train = pd.concat([y_train, X_train], axis=1)
    data_valid = pd.concat([y_valid, X_valid], axis=1)
    
    data_train.to_csv(train_metadata_file, index=False)
    data_valid.to_csv(valid_metadata_file, index=False)

if __name__ == "__main__":
    fire.Fire(split_train_val)
