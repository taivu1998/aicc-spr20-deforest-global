from .pretrained import *
from .models_3d import Sequential2DClassifier
from .fusion import FusionNet
from .seco import SeCoClassifier


def get_single_model(args):
    model_classes = globals().copy()
    model_name = args.get("model").split('-')[0]
    if model_name not in model_classes:
        raise ValueError(f"Unsupported model: {model_name}")
    model = model_classes[model_name](args)
    if args.get("late_fusion"):
        print("================Using FusionNet================")
        model = FusionNet(model, args)
    return model
