import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_launchpad_module():
    module_path = ROOT / "test_launchpad.py"
    spec = importlib.util.spec_from_file_location("launchpad_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_get_checkpoints_prefers_ckpts_directory(tmp_path):
    get_checkpoints = _load_launchpad_module().get_checkpoints
    expt_dir = tmp_path / "experiment"
    ckpt_dir = expt_dir / "ckpts"
    ckpt_dir.mkdir(parents=True)
    newest = ckpt_dir / "z.ckpt"
    oldest = ckpt_dir / "a.ckpt"
    newest.write_text("new")
    oldest.write_text("old")

    ckpts = get_checkpoints(expt_dir)

    assert ckpts[0] == newest


def test_get_checkpoints_raises_when_none_exist(tmp_path):
    get_checkpoints = _load_launchpad_module().get_checkpoints
    expt_dir = tmp_path / "experiment"
    expt_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        get_checkpoints(expt_dir)
