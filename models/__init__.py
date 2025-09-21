from .get_model import get_single_model
from .region import RegionModel
from .baseline import train_baseline


def get_model(model_args):
    if model_args.get("regions"):
        return RegionModel(model_args)
    return get_single_model(model_args)
