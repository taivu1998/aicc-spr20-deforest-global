import torch
import torchvision
from torch import nn


class SeCoClassifier(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.hparams = model_args
        self.num_classes = model_args["num_classes"]
        self.seco_ckpt_path = model_args["seco_ckpt_path"]

        if "resnet18" in self.seco_ckpt_path:
            pretrained_model = torchvision.models.resnet18()
        else:
            pretrained_model = torchvision.models.resnet50()

        self.num_ftrs = pretrained_model.fc.in_features
        self.backbone = nn.Sequential(
            *list(pretrained_model.children())[:-1],
            nn.Flatten()
        )
        self.backbone.load_state_dict(
            torch.load(self.seco_ckpt_path, map_location="cpu")
        )
        self.classifier = nn.Linear(self.num_ftrs, self.num_classes)

    def get_feature_dim(self):
        return self.num_ftrs

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, batch):
        inputs = batch['image']
        features = self.extract_features(inputs)
        logits = self.classifier(features)
        return logits
