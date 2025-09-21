import numpy as np
import torch
import torchvision.transforms as T
import imgaug.augmenters as iaa
from PIL import Image, ImageDraw

from util import constants as C
from .base_dataset import BaseDataset

class ClassificationDataset(BaseDataset):

    def __getitem__(self, index):
        image_info = self._image_info.iloc[index]
        event_index = self._image_info.index.values[index]
        image_arr = self._get_image(index)
        label = self._get_label(index)

        image_list = list()
        for image in image_arr:
                image_list.append(image)
        for t in self._transforms:
            if isinstance(t, iaa.Augmenter):
                if self._deterministic:
                    t = t.to_deterministic()
                for im_num, img in enumerate(image_list):
                    image_list[im_num] = t(image=img)
            elif isinstance(t, T.transforms.ToTensor):
                # necessary to avoid the negative stride
                for im_num, img in enumerate(image_list):
                    image_list[im_num] = t(np.ascontiguousarray(img))
            else:
                for im_num, img in enumerate(image_list):
                    image_list[im_num] = t(img)
        
        if not self._lrcn:
            output_image = torch.cat(image_list)
        else:
            output_image = torch.stack(image_list, 1)
        
        lat, lon = self._get_latlon(index)
        
        loss_areas = None
        if self._load_polygon_loss:
            loss_areas = image_info["loss_area"]/C.MAX_LOSS_AREA

        x_dict = {
            "image": output_image.squeeze(0),
            "index": index,
            "event_index": event_index, 
            "label": label,
            "lat": lat,
            "lon": lon,
            "region": image_info[C.REGION_HEADER],
            "gooder_id": image_info["GoodeR_ID"]
        }

        if self._load_polygon_loss:
            x_dict["loss_areas"] = loss_areas        

        if self._load_aux:
            aux_features = self._get_aux_features(index)
            x_dict.update(aux_features)

        if self._late_fusion_regions == 'onehot':
            region_embedding = C.REGION_EMBEDDINGS[x_dict["region"]]
            x_dict["region_embedding"] = torch.tensor(region_embedding, dtype=torch.float)

        return x_dict
