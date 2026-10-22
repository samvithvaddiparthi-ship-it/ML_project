"""Weakly-supervised localization: Class Activation Map -> bounding boxes.

Pipeline (as in the Stanford report, with two changes noted below):
  1. CAM  M(x, y) = sum_k w_k f_k(x, y) from the last conv map and FC weights.
  2. ReLU, bilinear up-sampling to image size, normalise by the map's maximum
     (a *relative* threshold -- the report used a fixed cut-off and noted that
     it should be dynamic).
  3. Threshold and find connected blobs.  The report used a hand-written DFS
     flood fill; `scipy.ndimage.label` computes exactly the same connected
     components, much faster.
  4. One box per blob (min/max x, y).  Tiny blobs are dropped and at most
     `max_boxes` are kept (RSNA images have at most 4 boxes).
The threshold is tuned on the validation set instead of fixed by hand.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_cams(model, ds, device, batch: int = 32, workers: int = 2):
    """Returns probs (N,) and low-resolution CAMs (N, h, w) as float16."""
    model.eval().to(device)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    probs, cams = [], []
    for x, _ in dl:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logit, cam = model.cam(x.to(device))
        probs.append(torch.sigmoid(logit.float()).cpu())
        cams.append(cam.float().cpu().half())
    return torch.cat(probs).numpy(), torch.cat(cams).numpy()


def upsample(cam: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(cam.astype(np.float32))[None, None]
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy()


def cam_to_boxes(cam: np.ndarray, out_h: int, out_w: int, thr: float, min_area_frac: float = 0.01,
                 max_boxes: int = 3, work_size: int = 256):
    """CAM (h, w) -> list of boxes (x1, y1, x2, y2) in out_h x out_w pixels + blob scores."""
    m = np.maximum(upsample(cam, work_size, work_size), 0)
    peak = m.max()
    if peak <= 0:
        return [], []
    m = m / peak
    lab, n = ndimage.label(m >= thr)
    boxes, scores = [], []
    if n == 0:
        return boxes, scores
    objs = ndimage.find_objects(lab)
    sx, sy = out_w / work_size, out_h / work_size
    for k, sl in enumerate(objs, start=1):
        area = (lab[sl] == k).sum()
        if area < min_area_frac * work_size * work_size:
            continue
        ys, xs = sl
        boxes.append([xs.start * sx, ys.start * sy, xs.stop * sx, ys.stop * sy])
        scores.append(float(m[sl][lab[sl] == k].mean()))
    order = np.argsort(scores)[::-1][:max_boxes]
    return [boxes[i] for i in order], [scores[i] for i in order]


def cam_predictions(probs, cams, sizes, thr: float, gate: float | None, min_area_frac: float,
                    max_boxes: int):
    """Turn CAMs into per-image predictions.  If `gate` is given, images whose
    classifier probability is below it get no box (end-to-end mode, as in the
    report); `gate=None` localizes every image (the report's 'oracle' mode)."""
    preds = []
    for p, cam, (h, w) in zip(probs, cams, sizes):
        if gate is not None and p < gate:
            preds.append(([], []))
            continue
        b, s = cam_to_boxes(cam, h, w, thr, min_area_frac, max_boxes)
        preds.append((b, [float(p) * si for si in s]))
    return preds
