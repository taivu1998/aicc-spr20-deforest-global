import pytest
import torch

from eval.loss import get_loss_fn


def test_get_loss_fn_ce():
    loss = get_loss_fn({"loss_fn": "CE"})
    assert isinstance(loss, torch.nn.CrossEntropyLoss)


def test_get_loss_fn_invalid():
    with pytest.raises(ValueError, match="not supported"):
        get_loss_fn({"loss_fn": "missing"})
