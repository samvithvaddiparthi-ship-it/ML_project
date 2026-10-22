"""Evaluation metrics.

Classification : accuracy, precision, recall (sensitivity), specificity, F1,
                 ROC-AUC, confusion matrix.
Localization   : (1) mean IoU as in the Stanford report: for every pneumonia
                     image, IoU between the union of predicted boxes and the
                     union of ground-truth boxes; an image with no predicted box
                     scores 0.
                 (2) the official RSNA Kaggle metric: mean over images of the
                     average (over IoU thresholds 0.40:0.05:0.75) of
                     TP / (TP + FP + FN).

All boxes are (x1, y1, x2, y2) in original-image pixels.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

RSNA_THRESHOLDS = np.arange(0.40, 0.76, 0.05)


# ---------------------------------------------------------------- classification
def classification_metrics(y_true, prob, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    try:
        auc = roc_auc_score(y_true, prob)
    except ValueError:
        auc = float("nan")
    return {"accuracy": (tp + tn) / len(y_true), "precision": prec, "recall": rec,
            "specificity": spec, "f1": f1, "auc": auc, "threshold": thr,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def best_f1_threshold(y_true, prob) -> float:
    """Threshold maximising F1 on (validation) data."""
    y_true = np.asarray(y_true).astype(int)
    grid = np.linspace(0.05, 0.95, 91)
    f1 = np.array([classification_metrics(y_true, prob, t)["f1"] for t in grid])
    ties = grid[f1 >= f1.max() - 1e-9]
    return float(np.median(ties))          # middle of the best range (robust to ties)


def roc_points(y_true, prob, n: int = 200):
    fpr, tpr, _ = roc_curve(y_true, prob)
    grid = np.linspace(0, 1, n)
    return grid.tolist(), np.interp(grid, fpr, tpr).tolist()


# ---------------------------------------------------------------- boxes
def xywh_to_xyxy(boxes):
    return [[x, y, x + w, y + h] for x, y, w, h in boxes]


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def union_mask_iou(pred, gt, h: int = 1024, w: int = 1024) -> float:
    """IoU between the union of predicted boxes and the union of GT boxes."""
    if len(gt) == 0 and len(pred) == 0:
        return 1.0
    if len(pred) == 0 or len(gt) == 0:
        return 0.0
    mp = np.zeros((h, w), bool)
    mg = np.zeros((h, w), bool)
    for x1, y1, x2, y2 in pred:
        mp[max(int(y1), 0):max(int(round(y2)), 0), max(int(x1), 0):max(int(round(x2)), 0)] = True
    for x1, y1, x2, y2 in gt:
        mg[max(int(y1), 0):int(round(y2)), max(int(x1), 0):int(round(x2))] = True
    inter = np.logical_and(mp, mg).sum()
    union = np.logical_or(mp, mg).sum()
    return float(inter / union) if union else 0.0


def rsna_image_score(pred_boxes, pred_scores, gt_boxes):
    """Kaggle RSNA metric for one image; returns None if the image is ignored
    (no ground truth and no prediction)."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return None
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0
    order = np.argsort(-np.asarray(pred_scores))
    preds = [pred_boxes[i] for i in order]
    ious = np.array([[box_iou(p, g) for g in gt_boxes] for p in preds])
    scores = []
    for t in RSNA_THRESHOLDS:
        matched = set()
        tp = 0
        for pi in range(len(preds)):
            best_j, best_iou = -1, t
            for gj in range(len(gt_boxes)):
                if gj not in matched and ious[pi, gj] > best_iou:
                    best_j, best_iou = gj, ious[pi, gj]
            if best_j >= 0:
                matched.add(best_j)
                tp += 1
        fp = len(preds) - tp
        fn = len(gt_boxes) - tp
        scores.append(tp / (tp + fp + fn))
    return float(np.mean(scores))


def localization_metrics(preds: list, gts: list, sizes: list) -> dict:
    """preds: list of (boxes, scores) per image; gts: list of GT box lists;
    sizes: list of (h, w).  Returns mean IoU (over positive images) and RSNA mAP
    (over all images that are not ignored)."""
    ious, rsna = [], []
    hit = 0
    for (pb, ps), gb, (h, w) in zip(preds, gts, sizes):
        s = rsna_image_score(pb, ps, gb)
        if s is not None:
            rsna.append(s)
        if len(gb):
            iou = union_mask_iou(pb, gb, h, w)
            ious.append(iou)
            hit += int(len(pb) > 0)
    n_pos = len(ious)
    return {"mean_iou": float(np.mean(ious)) if ious else float("nan"),
            "rsna_map": float(np.mean(rsna)) if rsna else float("nan"),
            "frac_pos_with_box": hit / max(n_pos, 1),
            "n_pos_images": n_pos, "n_scored_images": len(rsna),
            "iou_values": [round(v, 4) for v in ious]}
