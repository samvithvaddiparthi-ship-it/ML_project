"""Inference on a single X-ray: used by predict.py (CLI) and app.py (demo)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .cam import cam_to_boxes, upsample
from .data import to_tensor
from .models import build_classifier, build_detector
from .utils import get_device, load_json


class PneumoniaPredictor:
    """Loads the trained DenseNet-121 (+ optional Faster R-CNN) and the
    thresholds that were tuned on the validation set (from results.json)."""

    def __init__(self, out_dir: str | Path = "outputs", setting: str = "full", device=None):
        out_dir = Path(out_dir)
        self.device = device or get_device()
        res = load_json(out_dir / "results.json")
        cfg = res.get("config", {})
        self.cls_size = int(cfg.get("densenet_size", 384))
        self.det_size = int(cfg.get("det_size", 512))
        loc = res.get("localization", {}).get(setting, {})
        cam_cfg = loc.get("DenseNet121+CAM", {})
        self.cls_thr = float(cam_cfg.get("cls_gate",
                             res.get("classification", {}).get(setting, {}).get("DenseNet121", {})
                             .get("threshold", 0.5)))
        self.cam_thr = float(cam_cfg.get("cam_threshold", 0.5))
        gated = loc.get("DenseNet-gated FasterRCNN", {})
        self.det_thr = float(gated.get("score_threshold", loc.get("FasterRCNN", {}).get("score_threshold", 0.5)))
        self.min_area = float(cfg.get("cam_min_area_frac", 0.01))

        w = out_dir / "models" / f"DenseNet121_{setting}.pt"
        if not w.exists():
            raise FileNotFoundError(f"{w} not found - run training first or copy the weights there")
        self.clf = build_classifier("DenseNet121", pretrained=False)
        self.clf.load_state_dict(torch.load(w, map_location="cpu"))
        self.clf.eval().to(self.device)
        self.det = None
        dw = out_dir / "models" / "FasterRCNN.pt"
        if dw.exists():
            self.det = build_detector(pretrained=False, size=self.det_size)
            self.det.load_state_dict(torch.load(dw, map_location="cpu"))
            self.det.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, img: np.ndarray) -> dict:
        """img: uint8 grayscale, any size. Boxes are returned in its pixels."""
        h, w = img.shape[:2]
        x = cv2.resize(img, (self.cls_size, self.cls_size), interpolation=cv2.INTER_AREA)
        logit, cam = self.clf.cam(to_tensor(x, True)[None].to(self.device))
        prob = float(torch.sigmoid(logit)[0])
        cam = cam[0].float().cpu().numpy()
        cam_boxes, cam_scores = cam_to_boxes(cam, h, w, self.cam_thr, self.min_area, 3)
        heat = np.maximum(upsample(cam, h, w), 0)
        heat = heat / heat.max() if heat.max() > 0 else heat
        out = {"prob": prob, "positive": prob >= self.cls_thr, "cls_threshold": self.cls_thr,
               "heatmap": heat, "cam_boxes": cam_boxes if prob >= self.cls_thr else [],
               "det_boxes": [], "det_scores": []}
        if self.det is not None:
            xd = cv2.resize(img, (self.det_size, self.det_size), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(np.repeat((xd.astype(np.float32) / 255)[None], 3, 0)).to(self.device)
            r = self.det([t])[0]
            b, s = r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()
            keep = s >= self.det_thr
            b = b[keep] * np.array([w / self.det_size, h / self.det_size] * 2)
            if prob >= self.cls_thr:
                out["det_boxes"], out["det_scores"] = b.tolist(), s[keep].tolist()
        return out


def render(img: np.ndarray, pred: dict, gt_boxes=None) -> np.ndarray:
    """RGB visualisation: CAM heat-map, CAM boxes (blue), detector boxes (green),
    ground truth (red, optional)."""
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    heat = cv2.applyColorMap((pred["heatmap"] * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    vis = cv2.addWeighted(base, 0.65, heat, 0.35, 0)
    t = max(2, img.shape[0] // 250)
    for b in gt_boxes or []:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (227, 52, 47), t)
    for b in pred["cam_boxes"]:
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (47, 111, 227), t)
    for b, s in zip(pred["det_boxes"], pred["det_scores"]):
        cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (24, 165, 88), t)
        cv2.putText(vis, f"{s:.2f}", (int(b[0]), max(int(b[1]) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    img.shape[0] / 1400, (24, 165, 88), t)
    return vis
