import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F

from .get_model import get_single_model
from util import constants as C


class RegionModel(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.regions = C.REGIONS
        self.models = nn.ModuleDict({
            k: get_single_model(model_args) for k in self.regions
        })

    def forward(self, batch):
        image, region = batch['image'], batch['region']
        output = torch.cat(
            [self.models[C.REGIONS[r.item()]]({"image": x.unsqueeze(0)}) for x, r in zip(image, region)])
        return output
