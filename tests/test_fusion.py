import torch

from models.fusion import FusionNet


class DummyBackbone(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

    def get_feature_dim(self):
        return self.feature_dim

    def extract_features(self, x):
        batch_size = x.size(0)
        base = torch.arange(self.feature_dim, dtype=torch.float32)
        return base.unsqueeze(0).repeat(batch_size, 1)

    def forward(self, batch):
        raise NotImplementedError


def test_fusion_uses_backbone_feature_dim_dynamically():
    model = FusionNet(
        DummyBackbone(feature_dim=13),
        {
            "lrcn": False,
            "geo_embedding_dim": 5,
            "late_fusion_embedding_dim": 16,
            "late_fusion_dropout": 0.0,
            "cnn_logits_dim": 7,
            "late_fusion_regions": "onehot",
            "late_fusion_polygon_loss": True,
            "load_aux": False,
            "aux_subset": False,
            "num_classes": 5,
        },
    )

    batch = {
        "image": torch.randn(2, 3, 8, 8),
        "region_embedding": torch.eye(7, dtype=torch.float32)[:2],
        "loss_areas": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "lat": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "lon": torch.tensor([3.0, 4.0], dtype=torch.float32),
    }

    output = model(batch)

    assert output.shape == (2, 5)
    assert model.cnn2embs.in_features == 13


def test_fusion_rejects_lrcn_backbones():
    try:
        FusionNet(
            DummyBackbone(feature_dim=8),
            {
                "lrcn": True,
                "late_fusion_regions": "none",
                "late_fusion_polygon_loss": False,
                "load_aux": False,
                "aux_subset": False,
                "num_classes": 5,
            },
        )
    except ValueError as exc:
        assert "LRCN" in str(exc)
    else:
        raise AssertionError("FusionNet should reject LRCN models")
