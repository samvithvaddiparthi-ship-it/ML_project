"""End-to-end experiment pipeline.  Each stage reads what earlier stages wrote
to `cache_dir` / `out_dir`, so stages can be re-run on their own:

    prepare    labels -> one row per image, patient split, DICOM cache
    baselines  Logistic Regression / SVM / Random Forest on pixels
    cnn        Paper-style CNN and DenseNet-121 (+ CAMs for val/test)
    detector   Faster R-CNN (supervised boxes)
    localize   CAM -> boxes, threshold tuning on val, IoU + RSNA mAP on test
    figures    all plots in outputs/figures
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import baselines as B
from .cam import cam_predictions, compute_cams
from .config import Config
from .data import (CLASS_OPACITY, CLASSES, ClsDataset, DetDataset, build_cache, build_label_table,
                   load_cache, load_meta, make_split, setting_subset)
from .metrics import (best_f1_threshold, classification_metrics, localization_metrics, roc_points,
                      xywh_to_xyxy)
from .models import build_classifier, build_detector, count_params
from .train import predict_detector, predict_probs, train_classifier, train_detector
from .utils import ensure_dir, get_device, load_json, save_json, seed_everything, timer, update_results

PAPER_REFERENCE = {
    "source": "Huang, Cortes, Monam - Weakly Supervised Pneumonia Localization (CS229, 2018)",
    "classification_test_accuracy": {"LogReg": 0.7302, "SVM": 0.5818, "RandomForest": 0.83,
                                     "PaperCNN": 0.9247},
    "localization_test_iou": {"R-CNN (supervised)": 0.1266, "CNN+CAM (weak)": 0.1508,
                              "CNN+CAM if every positive is localized": 0.379},
    "input_size": 128, "split": "70/20/10", "setting": "Normal vs Lung Opacity (stage-1 data)",
}

CNN_MODELS = ("PaperCNN", "DenseNet121")


def _paths(cfg: Config):
    out = ensure_dir(cfg.out_dir)
    return out, ensure_dir(out / "models"), ensure_dir(out / "preds"), ensure_dir(out / "figures")


def _gt_and_sizes(df: pd.DataFrame):
    gts = [xywh_to_xyxy(b) for b in df["boxes"]]
    sizes = list(zip(df["orig_h"].astype(int), df["orig_w"].astype(int)))
    return gts, sizes


# =============================================================== prepare
def stage_prepare(cfg: Config):
    cache = ensure_dir(cfg.cache_dir)
    out, *_ = _paths(cfg)
    if (cache / "meta.csv").exists() and (cache / f"img{cfg.cache_size}.npy").exists():
        print("cache exists - skipping DICOM decoding")
        df = load_meta(cache)
    else:
        df = build_label_table(cfg.data_dir)
        if cfg.limit:
            df = df.groupby("cls", group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(2, int(cfg.limit * len(g) / len(df)))),
                                   random_state=cfg.seed)).reset_index(drop=True)
        df = make_split(df, cfg.split_fracs, cfg.seed)
        with timer("decode_dicoms", out):
            df = build_cache(df, cfg.data_dir, cache, cfg.cache_size, cfg.num_workers)
    stats = {
        "n_images": int(len(df)),
        "n_boxes": int(df["n_boxes"].sum()),
        "class_counts": df["cls"].value_counts().to_dict(),
        "split_counts": {s: df[df.split == s]["cls"].value_counts().reindex(CLASSES).fillna(0)
                         .astype(int).to_dict() for s in ("train", "val", "test")},
        "boxes_per_positive": df[df.target == 1]["n_boxes"].value_counts().sort_index().to_dict(),
        "image_size": f"{int(df.orig_h.mode()[0])}x{int(df.orig_w.mode()[0])}",
    }
    pos = df[df.target == 1]
    wh = np.array([[b[2], b[3], h * w] for bs, h, w in zip(pos["boxes"], pos["orig_h"], pos["orig_w"])
                   for b in bs]) if len(pos) else np.zeros((0, 3))
    if len(wh):
        stats["box_w_mean"], stats["box_h_mean"] = float(wh[:, 0].mean()), float(wh[:, 1].mean())
        stats["box_area_frac_mean"] = float((wh[:, 0] * wh[:, 1] / wh[:, 2]).mean())
    for s in cfg.settings:
        sub = setting_subset(df, s)
        stats[f"positive_rate_{s}"] = float(sub.target.mean())
        stats[f"n_{s}"] = int(len(sub))
    update_results(out, ["dataset"], stats)
    update_results(out, ["paper_reference"], PAPER_REFERENCE)
    update_results(out, ["config"], cfg.to_dict())
    print(stats)
    return df


# =============================================================== baselines
def stage_baselines(cfg: Config):
    out, *_ = _paths(cfg)
    df = load_meta(cfg.cache_dir)
    arr = load_cache(cfg.cache_dir, cfg.cache_size)
    for setting in cfg.settings:
        sub = setting_subset(df, setting)
        tr, te = sub[sub.split == "train"], sub[sub.split == "test"]
        if len(tr) > cfg.baseline_max_train:
            tr = tr.sample(cfg.baseline_max_train, random_state=cfg.seed)
        with timer(f"baseline_features_{setting}", out):
            Xtr = B.flatten_images(arr, tr["idx"].values, cfg.baseline_size)
            Xte = B.flatten_images(arr, te["idx"].values, cfg.baseline_size)
        n_pca = min(cfg.baseline_pca, Xtr.shape[0] - 1, Xtr.shape[1])
        for name, model in B.build_baselines(n_pca, cfg.seed).items():
            with timer(f"baseline_{name}_{setting}", out):
                model.fit(Xtr, tr["target"].values)
                s_tr, s_te = B.scores(model, Xtr), B.scores(model, Xte)
            m = classification_metrics(te["target"].values, s_te, 0.5)
            m["train_accuracy"] = float(((s_tr >= 0.5) == tr["target"].values).mean())
            m["roc"] = roc_points(te["target"].values, s_te)
            update_results(out, ["classification", setting, name], m)
            print(f"{setting:5s} {name:12s} test acc {m['accuracy']:.4f}  auc {m['auc']:.4f}")


# =============================================================== CNNs
def _cls_datasets(cfg, sub, name):
    if name == "PaperCNN":
        size, three = cfg.paper_cnn_size, False
    else:
        size, three = cfg.densenet_size, True
    mk = lambda d, aug: ClsDataset(d, cfg.cache_dir, cfg.cache_size, size, three, aug)
    return (mk(sub[sub.split == "train"], True), mk(sub[sub.split == "val"], False),
            mk(sub[sub.split == "test"], False))


def stage_cnn(cfg: Config, models=CNN_MODELS):
    out, mdir, pdir, _ = _paths(cfg)
    device = get_device()
    df = load_meta(cfg.cache_dir)
    for setting in cfg.settings:
        sub = setting_subset(df, setting)
        for name in models:
            seed_everything(cfg.seed)
            tr, va, te = _cls_datasets(cfg, sub, name)
            model = build_classifier(name, cfg.pretrained)
            ep, lr, bs = ((cfg.paper_cnn_epochs, cfg.paper_cnn_lr, cfg.paper_cnn_batch) if name == "PaperCNN"
                          else (cfg.densenet_epochs, cfg.densenet_lr, cfg.densenet_batch))
            print(f"== {name} | setting={setting} | train {len(tr)} val {len(va)} test {len(te)} | "
                  f"{count_params(model)/1e6:.2f}M params | device {device}")
            with timer(f"train_{name}_{setting}", out):
                model, hist = train_classifier(model, tr, va, device, ep, lr, bs, cfg.num_workers)
            torch.save(model.state_dict(), mdir / f"{name}_{setting}.pt")
            pv, cv = compute_cams(model, va, device, bs, cfg.num_workers)
            pt, ct = compute_cams(model, te, device, bs, cfg.num_workers)
            ptr = predict_probs(model, tr.__class__(sub[sub.split == "train"], cfg.cache_dir, cfg.cache_size,
                                                    tr.size, tr.three, False), device, bs, cfg.num_workers)
            thr = best_f1_threshold(va.y, pv)
            m = classification_metrics(te.y, pt, thr)
            m["at_0.5"] = {k: v for k, v in classification_metrics(te.y, pt, 0.5).items() if k != "confusion"}
            m["train_accuracy"] = float(((ptr >= thr) == tr.y).mean())
            m["val"] = {k: v for k, v in classification_metrics(va.y, pv, thr).items() if k != "confusion"}
            m["roc"] = roc_points(te.y, pt)
            m["history"] = hist
            m["params_M"] = round(count_params(model) / 1e6, 2)
            m["input_size"] = tr.size
            update_results(out, ["classification", setting, name], m)
            np.savez_compressed(pdir / f"{name}_{setting}.npz",
                                val_idx=sub[sub.split == "val"]["idx"].values, val_prob=pv, val_cam=cv,
                                test_idx=sub[sub.split == "test"]["idx"].values, test_prob=pt, test_cam=ct,
                                cls_threshold=thr)
            print(f"{setting} {name}: test acc {m['accuracy']:.4f} auc {m['auc']:.4f} "
                  f"recall {m['recall']:.4f} (thr {thr:.2f})")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


# =============================================================== detector
def stage_detector(cfg: Config):
    out, mdir, pdir, _ = _paths(cfg)
    device = get_device()
    df = load_meta(cfg.cache_dir)
    seed_everything(cfg.seed)
    tr = df[df.split == "train"]
    ds_pos = DetDataset(tr[tr.target == 1], cfg.cache_dir, cfg.cache_size, cfg.det_size, True)
    ds_neg = DetDataset(tr[tr.target == 0], cfg.cache_dir, cfg.cache_size, cfg.det_size, True)
    model = build_detector(cfg.pretrained, cfg.det_size)
    print(f"== Faster R-CNN | pos {len(ds_pos)} neg pool {len(ds_neg)} | {count_params(model)/1e6:.1f}M params")
    with timer("train_detector", out):
        model, hist = train_detector(model, ds_pos, ds_neg, device, cfg.det_epochs, cfg.det_lr,
                                     cfg.det_batch, cfg.det_neg_ratio, cfg.num_workers)
    torch.save(model.state_dict(), mdir / "FasterRCNN.pt")
    preds = {}
    for split in ("val", "test"):
        d = df[df.split == split].reset_index(drop=True)
        with timer(f"detector_predict_{split}", out):
            raw = predict_detector(model, DetDataset(d, cfg.cache_dir, cfg.cache_size, cfg.det_size, False),
                                   device, cfg.det_batch, cfg.num_workers)
        sc = (d["orig_w"].values / cfg.det_size, d["orig_h"].values / cfg.det_size)
        for k, (b, s) in enumerate(raw):
            b = b.copy()
            b[:, [0, 2]] *= sc[0][k]
            b[:, [1, 3]] *= sc[1][k]
            preds[int(d["idx"].iloc[k])] = (b, s)
    with open(pdir / "FasterRCNN.pkl", "wb") as f:
        pickle.dump(preds, f)
    update_results(out, ["detector_training"], {"history": hist, "params_M": round(count_params(model) / 1e6, 1),
                                                "input_size": cfg.det_size, "epochs": cfg.det_epochs})


# =============================================================== localization
def _filter(preds, thr):
    out = []
    for b, s in preds:
        b, s = np.asarray(b).reshape(-1, 4), np.asarray(s)
        k = s >= thr
        out.append((b[k].tolist(), s[k].tolist()))
    return out


def _strip(m):
    return {k: v for k, v in m.items() if k != "iou_values"}


def stage_localize(cfg: Config):
    out, _, pdir, _ = _paths(cfg)
    df = load_meta(cfg.cache_dir).set_index("idx", drop=False)
    det = None
    if (pdir / "FasterRCNN.pkl").exists():
        with open(pdir / "FasterRCNN.pkl", "rb") as f:
            det = pickle.load(f)
    saved = {}
    for setting in cfg.settings:
        res = {}
        for name in CNN_MODELS:
            f = pdir / f"{name}_{setting}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            gate = float(z["cls_threshold"])
            dv, dt = df.loc[z["val_idx"]], df.loc[z["test_idx"]]
            gv, sv = _gt_and_sizes(dv)
            gt, st = _gt_and_sizes(dt)
            # tune the CAM threshold on validation (paper metric: mean IoU, end-to-end)
            sweep = {}
            for thr in cfg.cam_thresholds:
                pv = cam_predictions(z["val_prob"], z["val_cam"], sv, thr, gate, cfg.cam_min_area_frac,
                                     cfg.cam_max_boxes)
                sweep[thr] = _strip(localization_metrics(pv, gv, sv))
            best = max(sweep, key=lambda t: sweep[t]["mean_iou"])
            p_e2e = cam_predictions(z["test_prob"], z["test_cam"], st, best, gate, cfg.cam_min_area_frac,
                                    cfg.cam_max_boxes)
            p_orc = cam_predictions(z["test_prob"], z["test_cam"], st, best, None, cfg.cam_min_area_frac,
                                    cfg.cam_max_boxes)
            e2e, orc = localization_metrics(p_e2e, gt, st), localization_metrics(p_orc, gt, st)
            res[f"{name}+CAM"] = {"type": "weakly supervised", "cam_threshold": best, "cls_gate": gate,
                                  "val_sweep": {str(k): v for k, v in sweep.items()},
                                  "test": _strip(e2e), "test_oracle": _strip(orc),
                                  "test_iou_values": e2e["iou_values"]}
            saved[(setting, f"{name}+CAM")] = dict(zip(z["test_idx"].tolist(), p_e2e))
            print(f"{setting} {name}+CAM thr {best}: mIoU {e2e['mean_iou']:.4f} (oracle {orc['mean_iou']:.4f})"
                  f" RSNA mAP {e2e['rsna_map']:.4f}")

        if det is not None:
            sub = setting_subset(df, setting)
            dv, dt = sub[sub.split == "val"], sub[sub.split == "test"]
            gv, sv = _gt_and_sizes(dv)
            gt, st = _gt_and_sizes(dt)
            rv = [det[i] for i in dv["idx"]]
            rt = [det[i] for i in dt["idx"]]
            sweep = {t: _strip(localization_metrics(_filter(rv, t), gv, sv)) for t in cfg.det_score_thresholds}
            best = max(sweep, key=lambda t: sweep[t]["rsna_map"])
            p_det = _filter(rt, best)
            m = localization_metrics(p_det, gt, st)
            res["FasterRCNN"] = {"type": "supervised", "score_threshold": best,
                                 "val_sweep": {str(k): v for k, v in sweep.items()},
                                 "test": _strip(m), "test_iou_values": m["iou_values"]}
            saved[(setting, "FasterRCNN")] = dict(zip(dt["idx"].tolist(), p_det))
            print(f"{setting} FasterRCNN score>={best}: mIoU {m['mean_iou']:.4f} RSNA mAP {m['rsna_map']:.4f}")

            # classifier-gated detector: keep boxes only if DenseNet says "pneumonia"
            f = pdir / f"DenseNet121_{setting}.npz"
            if f.exists():
                z = np.load(f)
                prob_v = dict(zip(z["val_idx"].tolist(), z["val_prob"]))
                prob_t = dict(zip(z["test_idx"].tolist(), z["test_prob"]))
                best_m, best_cfg = -1, None
                for g in cfg.gate_thresholds:
                    for t in cfg.det_score_thresholds:
                        pv = [(b, s) if prob_v[i] >= g else ([], []) for i, (b, s) in
                              zip(dv["idx"], _filter(rv, t))]
                        mv = localization_metrics(pv, gv, sv)["rsna_map"]
                        if mv > best_m:
                            best_m, best_cfg = mv, (g, t)
                g, t = best_cfg
                pt = [(b, s) if prob_t[i] >= g else ([], []) for i, (b, s) in zip(dt["idx"], _filter(rt, t))]
                m = localization_metrics(pt, gt, st)
                res["DenseNet-gated FasterRCNN"] = {"type": "supervised + classifier gate", "gate": g,
                                                    "score_threshold": t, "val_rsna_map": best_m,
                                                    "test": _strip(m), "test_iou_values": m["iou_values"]}
                saved[(setting, "DenseNet-gated FasterRCNN")] = dict(zip(dt["idx"].tolist(), pt))
                print(f"{setting} gated detector (gate {g}, score {t}): mIoU {m['mean_iou']:.4f} "
                      f"RSNA mAP {m['rsna_map']:.4f}")
        update_results(out, ["localization", setting], res)
    with open(pdir / "localization_test_predictions.pkl", "wb") as f:
        pickle.dump(saved, f)


# =============================================================== demo samples
def stage_demo(cfg: Config, n_pos: int = 6, n_neg: int = 3):
    """Export a few *test-set* images as PNG (+ ground truth) for the live demo."""
    import cv2
    import json
    out, *_ = _paths(cfg)
    ddir = ensure_dir(out / "demo_samples")
    df = load_meta(cfg.cache_dir)
    arr = load_cache(cfg.cache_dir, cfg.cache_size)
    te = df[df.split == "test"]
    pick = pd.concat([te[te.target == 1].sample(min(n_pos, (te.target == 1).sum()), random_state=7),
                      te[te.cls == CLASSES[0]].sample(min(n_neg, (te.cls == CLASSES[0]).sum()), random_state=7),
                      te[te.cls == CLASSES[1]].sample(min(n_neg, (te.cls == CLASSES[1]).sum()), random_state=7)])
    gt = {}
    for _, r in pick.iterrows():
        name = f"{'pneumonia' if r.target else 'negative'}_{r.patientId}.png"
        cv2.imwrite(str(ddir / name), np.asarray(arr[r["idx"]]))
        s = cfg.cache_size / r["orig_w"]
        gt[name] = {"cls": r["cls"], "boxes": [[x * s, y * s, (x + w) * s, (y + h) * s] for x, y, w, h in r["boxes"]]}
    (ddir / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    print(f"exported {len(gt)} demo images to {ddir}")


# =============================================================== all
STAGES = {"prepare": stage_prepare, "baselines": stage_baselines, "cnn": stage_cnn,
          "detector": stage_detector, "localize": stage_localize, "demo": stage_demo}


def run(cfg: Config, stages):
    seed_everything(cfg.seed)
    out = ensure_dir(cfg.out_dir)
    cfg.save(out / "config.json")
    for s in stages:
        if s == "detector" and not cfg.run_detector:
            continue
        if s == "figures":
            from .viz import make_all_figures
            with timer("figures"):
                make_all_figures(cfg)
            continue
        with timer(f"stage_{s}", out):
            STAGES[s](cfg)
