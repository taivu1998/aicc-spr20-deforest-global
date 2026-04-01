import torch
import torch.nn as nn

from .pretrained import (DenseNet121,
                         DenseNet161,
                         DenseNet201,
                         ResNet18,
                         ResNet34,
                         ResNet101,
                         ResNet152)

cnn_dict = {'densenet121': DenseNet121,
            'densenet161': DenseNet161,
            'densenet201': DenseNet201,
            'resnet18': ResNet18,
            'resnet34': ResNet34,
            'resnet101': ResNet101,
            'resnet152': ResNet152,
            }


class Sequential2DClassifier(nn.Module):
    """LRCN Model.
    LRCN consists of a CNN whose outputs are fed into a stack of LSTMs. Both the CNN and
    LSTM weights are shared across time, so the model scales to arbitrarily long inputs.
    Based on the paper:
    "Long-term Recurrent Convolutional Networks for Visual Recognition and Description"
    by Jeff Donahue, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan,
    Sergio Guadarrama, Kate Saenko, Trevor Darrell
    (https://arxiv.org/abs/1411.4389).
    """

    def __init__(self, model_args=None):
        super().__init__()
        model_args['backbone'] = model_name = model_args["model"].split(
            '-')[1].lower()
        self.model = cnn_dict[model_name](model_args)
        self.hidden_dim = model_args["hidden_dim"]
        self.num_lstm_layers = model_args["num_lstm_layers"]
        self.num_classes = model_args["num_classes"]

        if model_args["composite"]:
            self.num_slices = 1
        elif model_args["first_last"]:
            self.num_slices = 2
        else:
            self.num_slices = 4

        self.num_ftrs = self.model.get_feature_dim()
        self.lstm = nn.LSTM(self.num_ftrs,
                            self.hidden_dim,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)
        self.classifier = nn.Linear(self.hidden_dim,
                                    self.num_classes)

    def forward(self, batch):
        inputs = batch['image']
        B, C, S, H, W = inputs.shape
        inputs = torch.transpose(inputs, 1, 2).contiguous()
        inputs = inputs.view(B * S, C, H, W)

        features = self.model.extract_features(inputs)
        features = features.view(B, S, self.num_ftrs)

        lstm_out, _ = self.lstm(features)
        final_outputs = lstm_out[:, -1, :]

        logits = self.classifier(final_outputs)
        return logits
