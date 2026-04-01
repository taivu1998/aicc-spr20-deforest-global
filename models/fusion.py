import torch
from torch import nn
from util import constants as C

class FusionNet(nn.Module):
    def __init__(self, model, model_args):
        super().__init__()
        self.hparams = model_args
        self.model = model

        if self.hparams.get("lrcn"):
            raise ValueError("Late fusion is not supported with LRCN models.")
        if not hasattr(self.model, "extract_features") or not hasattr(self.model, "get_feature_dim"):
            raise ValueError(
                "Late fusion requires a backbone with extract_features() "
                "and get_feature_dim()."
            )

        self.N = self.hparams.get("geo_embedding_dim", 5)
        self.embedding_dim = self.hparams.get("late_fusion_embedding_dim", 128)
        self.dropout_rate = self.hparams.get("late_fusion_dropout", 0.2)
        self.max_wavelength = self.hparams.get("geo_embedding_wave_length", 10)
        self.cnn_logits_dim = self.hparams.get("cnn_logits_dim", 128)
        self.polygon_loss = self.hparams.get("late_fusion_polygon_loss", False)
        self.use_aux = self.hparams.get("load_aux", False)
        self.aux_subset = self.hparams.get("aux_subset", False)
        self.late_fusion_regions = self.hparams["late_fusion_regions"]
        num_logit_features = self.cnn_logits_dim

        if self.late_fusion_regions == 'none':
            num_loc_features = 0
        elif self.late_fusion_regions == 'latlon':
            num_loc_features = self.N * 4
        elif self.late_fusion_regions == 'onehot':
            num_loc_features = len(C.REGIONS)
        else:
            raise ValueError(
                f"Invalid late_fusion_regions type entered: {self.late_fusion_regions}"
            )

        num_polygon_loss_features = 1 if self.polygon_loss else 0
        if self.aux_subset:
            self.aux_features = C.AUX_SUBSET_FEATURE_HEADER
        else:
            self.aux_features = C.AUX_FEATURE_HEADER if self.use_aux else []

        num_aux_features = self.use_aux * len(self.aux_features)
        self.input_dim = (
            num_logit_features
            + num_loc_features
            + num_polygon_loss_features
            + num_aux_features
        )
        self.output_dim = self.hparams["num_classes"]
        self.cnn2embs = nn.Linear(self.model.get_feature_dim(), num_logit_features)
        self.fc = nn.Sequential(
                    nn.Linear(self.input_dim, self.embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.embedding_dim, self.embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(self.embedding_dim, self.output_dim)
                )
            
    def latlon_encoding(self, lat, lon):
        lat = self.positional_encoding(
            lat, C.LAT_MIN, C.LAT_MAX)
        lon = self.positional_encoding(
            lon, C.LON_MIN, C.LON_MAX)
        return torch.cat([lat, lon], dim=-1)
        
    def positional_encoding(self, value, value_min, value_max):
        """
        Geo/positional encoding following:
        1. Attention is all you need: https://arxiv.org/pdf/1706.03762.pdf
        2. Improving Urban-Scene Segmentation via
        Height-driven Attention Networks: https://arxiv.org/abs/2003.05128.pdf
        """

        value = (value - value_min) / (value_max - value_min)
        encoding = []
        for i in range(self.N):
            s = torch.sin(value / (self.max_wavelength ** (i / self.N)))
            c = torch.cos(value / (self.max_wavelength ** (i / self.N)))
            encoding.append(s)
            encoding.append(c)
        return torch.stack(encoding, dim=-1)

    def get_feature_dim(self):
        return self.cnn_logits_dim

    def extract_features(self, x):
        return self.model.extract_features(x)

    def extract_pre_logits(self, batch):
        return self.model.extract_features(batch['image'])

    def get_logits(self, batch):
        x = self.extract_features(batch['image'])
        x = self.cnn2embs(x.view(x.size(0), -1))
        return x

    def forward(self, batch):
        logits = self.get_logits(batch)
        fusion_feats = [logits.float()]

        for feat_name in self.aux_features:
            feat = batch.get(feat_name)
            feat = feat.unsqueeze(1)
            fusion_feats.append(feat.float())

        if self.late_fusion_regions == 'latlon':
            lat, lon = batch['lat'], batch['lon']
            latlon_encoding = self.latlon_encoding(lat, lon)
            fusion_feats.append(latlon_encoding.float())
        elif self.late_fusion_regions == 'onehot':
            fusion_feats.append(batch['region_embedding'].float())

        if self.polygon_loss:
            loss_areas = batch['loss_areas']
            fusion_feats.append(loss_areas.float().unsqueeze(1))

        fusion = torch.cat(fusion_feats, dim=1)
        y_logit_cls = self.fc(fusion)
        return y_logit_cls
