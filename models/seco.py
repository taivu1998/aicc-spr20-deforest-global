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
        pretrained_model = nn.Sequential(*list(pretrained_model.children())[:-1], nn.Flatten())
        pretrained_model.load_state_dict(torch.load(self.seco_ckpt_path))
        
        self.backbone = pretrained_model
        self.num_ftrs = list(self.backbone.children())[-3][1].bn2.num_features
        self.classifier = nn.Linear(self.num_ftrs, self.num_classes)

    def forward(self, batch):
        inputs = batch['image']
        features = self.backbone(inputs)
        logits = self.classifier(features)
        return logits