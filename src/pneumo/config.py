"""Central configuration for every stage of the pipeline.

All hyper-parameters live here so that the README, the write-up and the code
agree. `Config.fast()` returns a reduced configuration (fewer epochs) for a
quick end-to-end run; `Config.smoke()` is a tiny CPU configuration used by the
synthetic-data smoke test.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---------------- paths ----------------
    data_dir: str = "/kaggle/input/rsna-pneumonia-detection-challenge"
    cache_dir: str = "cache"          # pre-processed image arrays (not committed)
    out_dir: str = "outputs"          # results.json, figures, model weights

    # ---------------- data ----------------
    cache_size: int = 512             # every DICOM is resized once to this size
    split_fracs: tuple = (0.70, 0.15, 0.15)   # train / val / test, patient level
    seed: int = 42
    limit: int | None = None          # use only N images (debugging)
    num_workers: int = 4

    # ---------------- classical baselines (LR / SVM / RF) ----------------
    baseline_size: int = 64           # images flattened at 64x64 = 4096 features
    baseline_pca: int = 150
    baseline_max_train: int = 12000   # SVM is O(n^2); cap the train set

    # ---------------- paper-style CNN (replication) ----------------
    paper_cnn_size: int = 128         # the paper compressed 1024 -> 128
    paper_cnn_epochs: int = 20        # paper: Adam, lr 1e-4, 20 epochs
    paper_cnn_lr: float = 1e-4
    paper_cnn_batch: int = 64

    # ---------------- DenseNet-121 transfer learning (improvement) ----------------
    densenet_size: int = 384
    densenet_epochs: int = 10
    densenet_lr: float = 3e-4
    densenet_batch: int = 32
    pretrained: bool = True           # ImageNet weights (needs internet once)

    # ---------------- CAM -> boxes ----------------
    cam_thresholds: tuple = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    cam_min_area_frac: float = 0.01   # drop blobs smaller than 1% of the image
    cam_max_boxes: int = 3

    # ---------------- Faster R-CNN (supervised localization) ----------------
    det_size: int = 512
    det_epochs: int = 6
    det_lr: float = 0.01
    det_batch: int = 8
    det_neg_ratio: float = 1.0        # negatives sampled per positive each epoch
    det_score_thresholds: tuple = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    gate_thresholds: tuple = (0.2, 0.3, 0.4, 0.5, 0.6)

    # which experiments to run
    settings: tuple = ("paper", "full")
    run_detector: bool = True

    # ------------------------------------------------------------------
    @staticmethod
    def fast(**kw) -> "Config":
        """~45 min on a Kaggle T4: fewer epochs, same models."""
        base = dict(paper_cnn_epochs=8, densenet_epochs=4, densenet_size=320, det_epochs=2)
        return Config(**{**base, **kw})

    @staticmethod
    def smoke(**kw) -> "Config":
        """Tiny CPU config for the synthetic-data smoke test."""
        base = dict(cache_size=128, baseline_size=32, baseline_pca=10,
                    paper_cnn_size=64, paper_cnn_epochs=2, paper_cnn_batch=16,
                    densenet_size=96, densenet_epochs=1, densenet_batch=8,
                    det_size=128, det_epochs=1, det_batch=4, pretrained=False, num_workers=0)
        return Config(**{**base, **kw})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> "Config":
        d = json.loads(Path(path).read_text())
        for k, v in list(d.items()):
            if isinstance(v, list):
                d[k] = tuple(v)
        return Config(**d)
