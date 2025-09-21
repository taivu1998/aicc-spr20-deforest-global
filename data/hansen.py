import pandas
import os
import logging
import numpy
from pathlib import Path

from util import constants as C
from .classification_dataset import ClassificationDataset


class HansenDriversDataset(ClassificationDataset):

    DATA_SPLIT_TO_META = {
        C.TRAIN_SPLIT: C.HANSEN_TRAIN_V5_PATH_SHORT,
        C.VAL_SPLIT: C.HANSEN_VAL_V5_PATH_SHORT,
        C.TEST_SPLIT: C.HANSEN_TEST_V5_PATH_SHORT
    }

    def process_file(self):
        """This function processes image meta-CSV into a list of
        (label, lat, long, image path, year) tuples.
        """

        meta_filename = Path(self._image_path) / \
            self.DATA_SPLIT_TO_META[self._data_split]
        self._image_info = pandas.read_csv(meta_filename,
                                           header=0,
                                           index_col=0,
                                           na_filter=False)
        self.filter_examples()

    def filter_examples(self):
        # NOTE: right now we drop images that don't exist
        # from the CSV (image_path == None). We need to go
        # retrieve these.
        # NOTE: we also drop labels 'Uncertain/Other' for now.

        valid_img_paths = self._image_info[C.IMG_PATH_HEADER] != 'None'
        valid_labels = ~self._image_info[C.LABEL_HEADER].isin(
            C.HANSEN_IGNORED_LABEL_IDXS)

        logging.debug(f'Original number of samples: {len(self._image_info)}')
        logging.debug(f'Keeping examples from region(s): {self._regions}')
        print(f'Number of all samples from {self._data_split} split: {len(self._image_info)}')

        if (self._year_cutoff is None) or (self._data_split != C.TRAIN_SPLIT): 
            self._image_info = self._image_info[valid_img_paths & valid_labels]
        else:
            valid_years = self._image_info[C.YEAR_HEADER] >= self._year_cutoff
            self._image_info = self._image_info[valid_img_paths & valid_labels & valid_years]
        logging.debug(f'Number of samples retained: {len(self._image_info)}')
        print(f'Number of samples retained from {self._data_split} split: {len(self._image_info)}')
        
        