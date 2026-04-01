from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import util.constants as C
from data.base_dataset import BaseDataset


class DummyDataset(BaseDataset):
    def process_file(self):
        raise NotImplementedError


def _make_dataset(tmp_path):
    event_dir = tmp_path / "event_1"
    rgb_dir = event_dir / "rgb"
    rgb_dir.mkdir(parents=True)

    composite_path = event_dir / "composite.png"
    Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(composite_path)

    dataset = DummyDataset.__new__(DummyDataset)
    dataset._image_info = pd.DataFrame([
        {
            C.IMG_PATH_HEADER: str(event_dir),
            C.IMG_OPTION_COMPOSITE: "composite.png",
            C.LABEL_HEADER: 0,
            C.LATITUDE_HEADER: 0.0,
            C.LONGITUDE_HEADER: 0.0,
            C.YEAR_HEADER: 2018,
            C.REGION_HEADER: 0,
            "loss_area": 1.0,
            "GoodeR_ID": 123,
        }
    ])
    dataset._lrcn = True
    dataset._load_mode = "annual"
    dataset._first_last = False
    dataset._padding = "end"
    dataset._img_option = C.IMG_OPTION_COMPOSITE
    dataset._transforms = []
    dataset._deterministic = True
    dataset._load_polygon_loss = False
    dataset._load_aux = False
    dataset._regions = None
    dataset._late_fusion_regions = "none"
    dataset._data_split = C.TRAIN_SPLIT
    return dataset


def test_lrcn_missing_sequence_falls_back_to_event_composite(tmp_path):
    dataset = _make_dataset(tmp_path)

    images = dataset._get_image(0)

    assert len(images) == C.MAX_IMGS_PER_LOCATION
    assert images[0].shape == (8, 8, 3)
    assert np.any(images[0] != 0)
    for blank in images[1:]:
        assert np.all(blank == 0)
