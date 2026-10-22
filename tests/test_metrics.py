"""Unit tests for the metric and CAM-to-box code (run: pytest -q)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pneumo.cam import cam_to_boxes  # noqa: E402
from pneumo.metrics import (box_iou, classification_metrics, localization_metrics,  # noqa: E402
                            rsna_image_score, union_mask_iou)


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert abs(box_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9


def test_union_mask_iou():
    assert union_mask_iou([], [[0, 0, 10, 10]], 32, 32) == 0.0
    assert union_mask_iou([[0, 0, 10, 10]], [[0, 0, 10, 10]], 32, 32) == 1.0
    # two GT boxes, one prediction covering exactly one of them -> IoU 0.5
    assert abs(union_mask_iou([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30]], 32, 32) - 0.5) < 1e-9


def test_rsna_metric():
    gt = [[0, 0, 100, 100]]
    assert rsna_image_score([], [], []) is None                      # ignored
    assert rsna_image_score([[0, 0, 100, 100]], [0.9], []) == 0.0     # false positive on negative
    assert rsna_image_score([], [], gt) == 0.0                        # missed
    assert rsna_image_score([[0, 0, 100, 100]], [0.9], gt) == 1.0     # perfect
    # IoU 0.5 -> true positive for thresholds 0.40 and 0.45 only = 2/8
    s = rsna_image_score([[0, 0, 100, 50]], [0.9], gt)
    assert abs(s - 2 / 8) < 1e-9


def test_localization_metrics_aggregate():
    preds = [([[0, 0, 10, 10]], [0.9]), ([], [])]
    gts = [[[0, 0, 10, 10]], []]
    m = localization_metrics(preds, gts, [(32, 32), (32, 32)])
    assert m["mean_iou"] == 1.0 and m["rsna_map"] == 1.0 and m["n_scored_images"] == 1


def test_classification_metrics():
    m = classification_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert m["accuracy"] == 0.5 and m["recall"] == 0.5 and m["specificity"] == 0.5
    assert m["auc"] == 0.75


def test_cam_to_boxes_finds_two_blobs():
    cam = np.zeros((16, 16))
    cam[2:6, 2:6] = 1.0
    cam[10:14, 9:13] = 0.8
    boxes, scores = cam_to_boxes(cam, 1024, 1024, thr=0.5, min_area_frac=0.001, max_boxes=3)
    assert len(boxes) == 2 and scores[0] >= scores[1]
    x1, y1, x2, y2 = boxes[0]
    assert 64 < x1 < 192 and 64 < y1 < 192 and 320 < x2 < 448
