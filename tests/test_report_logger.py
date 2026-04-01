import numpy as np

from lightning.logger import ReportLogger


class DummyReporter(ReportLogger):
    def __init__(self, save_dir):
        self.hparams = {
            "labels": [
                "Commodity Driven Deforestation",
                "Shifting Agriculture",
                "Forestry",
                "Wildfire",
                "Urbanization",
            ],
            "default_save_path": str(save_dir),
        }


def test_report_logger_handles_numeric_region_ids(tmp_path):
    reporter = DummyReporter(tmp_path)

    reporter.report(
        preds=np.array([[0], [1], [2], [3], [4]]),
        labels=np.array([[0], [1], [2], [3], [4]]),
        regions=np.array([[0], [1], [0], [1], [0]]),
        areas=np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]),
    )

    assert (tmp_path / "class_table_na.txt").exists()
    assert (tmp_path / "class_table_la.txt").exists()
    assert (tmp_path / "conf_heatmap_overall.jpg").exists()
