import sys
import types

import torch


def _install_lightning_stubs():
    if "pytorch_lightning" not in sys.modules:
        pl_module = types.ModuleType("pytorch_lightning")

        class LightningModule(torch.nn.Module):
            def save_hyperparameters(self, params):
                self.hparams = params

        pl_module.LightningModule = LightningModule
        sys.modules["pytorch_lightning"] = pl_module

    if "ignite" not in sys.modules:
        ignite_module = types.ModuleType("ignite")
        metrics_module = types.ModuleType("ignite.metrics")

        class Accuracy:
            pass

        metrics_module.Accuracy = Accuracy
        ignite_module.metrics = metrics_module
        sys.modules["ignite"] = ignite_module
        sys.modules["ignite.metrics"] = metrics_module


_install_lightning_stubs()

import lightning.model as lightning_model
from lightning.model import Model


class DummyClassifier(torch.nn.Module):
    def forward(self, batch):
        batch_size = batch["label"].shape[0]
        return torch.zeros(batch_size, 5, dtype=torch.float32)


class DummyFusionLikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_logits_called = False
        self.get_logits_called = False

    def forward(self, batch):
        return torch.zeros(batch["label"].shape[0], 5, dtype=torch.float32)

    def extract_pre_logits(self, batch):
        self.pre_logits_called = True
        return torch.ones(batch["label"].shape[0], 9, dtype=torch.float32)

    def get_logits(self, batch):
        self.get_logits_called = True
        return torch.zeros(batch["label"].shape[0], 3, dtype=torch.float32)


class DummyDataset:
    def class_weights(self):
        return [1.0, 2.0, 3.0, 4.0, 5.0]


def test_model_initializes_with_class_weights(monkeypatch):
    monkeypatch.setattr(lightning_model, "get_model", lambda _: DummyClassifier())
    monkeypatch.setattr(Model, "get_dataset", lambda self, split: DummyDataset())

    params = {
        "output_pre_logits": False,
        "class_weight": True,
        "loss_fn": "CE",
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "lr_schedule": False,
        "dataset": ".",
        "regions": None,
        "train_img_option": "composite",
        "eval_img_option": "composite",
        "first_last": False,
        "lrcn": False,
        "load_polygon_loss": False,
        "late_fusion_regions": "none",
        "load_aux": False,
        "load_mode": "annual",
        "year_cutoff": None,
        "num_dl_workers": 0,
    }

    model = Model(params)

    assert isinstance(model.loss, torch.nn.CrossEntropyLoss)


def test_model_prefers_extract_pre_logits_over_get_logits(monkeypatch):
    dummy_model = DummyFusionLikeModel()
    monkeypatch.setattr(lightning_model, "get_model", lambda _: dummy_model)

    params = {
        "output_pre_logits": True,
        "class_weight": False,
        "loss_fn": "CE",
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "lr_schedule": False,
        "dataset": ".",
        "regions": None,
        "train_img_option": "composite",
        "eval_img_option": "composite",
        "first_last": False,
        "lrcn": False,
        "load_polygon_loss": False,
        "late_fusion_regions": "none",
        "load_aux": False,
        "load_mode": "annual",
        "year_cutoff": None,
        "num_dl_workers": 0,
    }

    model = Model(params)
    batch = {"image": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])}

    extracted = model._extract_pre_logits(batch)

    assert extracted.shape == (2, 9)
    assert dummy_model.pre_logits_called is True
    assert dummy_model.get_logits_called is False
