"""All figures for the write-up and the slides (saved to outputs/figures)."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .cam import upsample
from .config import Config
from .data import CLASSES, load_cache, load_meta
from .utils import ensure_dir, load_json

GT_C, CAM_C, DET_C = "#e3342f", "#2f6fe3", "#18a558"
MODEL_COLORS = {"LogReg": "#9aa0a6", "SVM": "#c49a6c", "RandomForest": "#8e6bbf",
                "PaperCNN": "#f29d38", "DenseNet121": "#2f6fe3"}
LABELS = {"LogReg": "Logistic Regression", "SVM": "SVM (RBF)", "RandomForest": "Random Forest",
          "PaperCNN": "Paper-style CNN (128px)", "DenseNet121": "DenseNet-121 (pretrained)"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _draw_boxes(ax, boxes, color, scale=1.0, lw=1.8, ls="-"):
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(patches.Rectangle((x1 * scale, y1 * scale), (x2 - x1) * scale, (y2 - y1) * scale,
                                       fill=False, edgecolor=color, linewidth=lw, linestyle=ls))


def fig_data_overview(cfg, df, arr, fdir):
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1, 1, 1], wspace=0.15)
    ax = fig.add_subplot(gs[0])
    counts = df["cls"].value_counts().reindex(CLASSES).fillna(0)
    ax.barh(["Normal", "Not normal,\nno opacity", "Lung opacity\n(pneumonia)"], counts.values,
            color=["#9aa0a6", "#c49a6c", GT_C])
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v):,}", va="center", fontsize=9)
    ax.set_xlabel("images")
    ax.set_title("RSNA training images by class")
    ax.set_xlim(0, counts.max() * 1.25)
    rng = np.random.default_rng(1)
    s = cfg.cache_size
    for j, c in enumerate(CLASSES):
        a = fig.add_subplot(gs[j + 1])
        cand = df[df.cls == c]
        if c == CLASSES[2]:
            cand = cand[cand.n_boxes >= 2] if (cand.n_boxes >= 2).any() else cand
        if not len(cand):
            a.axis("off")
            continue
        r = cand.iloc[rng.integers(len(cand))]
        a.imshow(arr[r["idx"]], cmap="gray")
        _draw_boxes(a, [[x, y, x + w, y + h] for x, y, w, h in r["boxes"]], GT_C, s / r["orig_w"])
        a.set_title(["Normal", "Not normal / no opacity", "Lung opacity + boxes"][j], fontsize=9)
        a.axis("off")
    fig.savefig(fdir / "data_overview.png")
    plt.close(fig)


def fig_roc(res, setting, fdir):
    cl = res.get("classification", {}).get(setting, {})
    if not cl:
        return
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for name in ["LogReg", "SVM", "RandomForest", "PaperCNN", "DenseNet121"]:
        if name in cl and "roc" in cl[name]:
            fpr, tpr = cl[name]["roc"]
            ax.plot(fpr, tpr, color=MODEL_COLORS[name], lw=2 if "CNN" in name or "Dense" in name else 1.4,
                    label=f"{LABELS[name]}  AUC {cl[name]['auc']:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="#bbb", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title({"paper": "ROC - Normal vs Pneumonia (paper setting)",
                  "full": "ROC - Pneumonia vs all (full dataset)"}[setting], fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.savefig(fdir / f"roc_{setting}.png")
    plt.close(fig)


def fig_training(res, fdir):
    cl = res.get("classification", {})
    runs = [(s, n) for s in cl for n in ("PaperCNN", "DenseNet121") if n in cl[s] and "history" in cl[s][n]]
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for s, n in runs:
        h = cl[s][n]["history"]
        ls = "-" if s == "paper" else "--"
        lab = f"{LABELS[n].split(' (')[0]} - {s}"
        axes[0].plot(h["epoch"], h["train_loss"], ls, color=MODEL_COLORS[n], label=lab)
        axes[1].plot(h["epoch"], h["val_auc"], ls, color=MODEL_COLORS[n], marker="o", ms=3, label=lab)
    axes[0].set_title("Training loss (weighted BCE)")
    axes[1].set_title("Validation ROC-AUC")
    for a in axes:
        a.set_xlabel("epoch")
    axes[1].legend(fontsize=7.5, frameon=False)
    fig.savefig(fdir / "training_curves.png")
    plt.close(fig)


def fig_confusion(res, setting, name, fdir):
    m = res.get("classification", {}).get(setting, {}).get(name)
    if not m:
        return
    c = m["confusion"]
    mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(3.2, 3))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["pred. no", "pred. pneumonia"])
    ax.set_yticks([0, 1], ["no pneumonia", "pneumonia"])
    ax.set_title(f"{LABELS[name].split(' (')[0]} - test ({setting})", fontsize=9)
    fig.savefig(fdir / f"confusion_{setting}_{name}.png")
    plt.close(fig)


def fig_paper_comparison(res, fdir):
    ref = res.get("paper_reference", {})
    cl = res.get("classification", {}).get("paper", {})
    loc = res.get("localization", {}).get("paper", {})
    if not cl:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    names = [n for n in ["LogReg", "SVM", "RandomForest", "PaperCNN", "DenseNet121"] if n in cl]
    x = np.arange(len(names))
    paper = [ref.get("classification_test_accuracy", {}).get(n, np.nan) for n in names]
    ours = [cl[n]["accuracy"] for n in names]
    axes[0].bar(x - 0.2, paper, 0.4, color="#cfd3d8", label="Stanford report")
    axes[0].bar(x + 0.2, ours, 0.4, color=[MODEL_COLORS[n] for n in names], label="ours")
    for xi, v in zip(x, ours):
        axes[0].text(xi + 0.2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x, paper):
        if not np.isnan(v):
            axes[0].text(xi - 0.2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color="#777")
    axes[0].set_xticks(x, [LABELS[n].split(" (")[0].replace(" ", "\n", 1) for n in names], fontsize=8)
    axes[0].set_ylim(0, 1.18)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Classification (Normal vs Pneumonia)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left", ncol=2)
    lab, val, col = ["Report\nR-CNN", "Report\nCNN\n+CAM"], [0.1266, 0.1508], ["#cfd3d8", "#cfd3d8"]
    for k, short, c in [("PaperCNN+CAM", "Ours\nCNN\n+CAM", MODEL_COLORS["PaperCNN"]),
                        ("DenseNet121+CAM", "Ours\nDenseNet\n+CAM", MODEL_COLORS["DenseNet121"]),
                        ("FasterRCNN", "Ours\nFaster\nR-CNN", DET_C),
                        ("DenseNet-gated FasterRCNN", "Ours\ngated\nF-RCNN", "#0d6b3a")]:
        if k in loc:
            lab.append(short)
            val.append(loc[k]["test"]["mean_iou"])
            col.append(c)
    axes[1].bar(range(len(val)), val, color=col)
    for i, v in enumerate(val):
        axes[1].text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks(range(len(val)), lab, fontsize=7.5)
    axes[1].set_ylabel("mean IoU on pneumonia test images")
    axes[1].set_ylim(0, max(val) * 1.2 + 1e-3)
    axes[1].set_title("Localization")
    fig.savefig(fdir / "paper_comparison.png")
    plt.close(fig)


def fig_localization_examples(cfg, df, arr, res, fdir, setting="paper", n=6):
    pdir = Path(cfg.out_dir) / "preds"
    f = pdir / f"DenseNet121_{setting}.npz"
    lp = pdir / "localization_test_predictions.pkl"
    if not f.exists() or not lp.exists():
        return
    z = np.load(f)
    with open(lp, "rb") as fh:
        saved = pickle.load(fh)
    cam_p = saved.get((setting, "DenseNet121+CAM"), {})
    det_p = saved.get((setting, "DenseNet-gated FasterRCNN"), saved.get((setting, "FasterRCNN"), {}))
    pos = [k for k, i in enumerate(z["test_idx"]) if df.loc[i, "target"] == 1]
    rng = np.random.default_rng(3)
    pick = rng.choice(pos, min(n, len(pos)), replace=False) if pos else []
    if not len(pick):
        return
    s = cfg.cache_size
    fig, axes = plt.subplots(len(pick), 3, figsize=(7.5, 2.55 * len(pick)))
    axes = np.atleast_2d(axes)
    for r, k in enumerate(pick):
        i = int(z["test_idx"][k])
        row = df.loc[i]
        img = arr[i]
        sc = s / row["orig_w"]
        gt = [[x, y, x + w, y + h] for x, y, w, h in row["boxes"]]
        cam = np.maximum(upsample(z["test_cam"][k], s, s), 0)
        cam = cam / cam.max() if cam.max() > 0 else cam
        axes[r, 0].imshow(img, cmap="gray")
        _draw_boxes(axes[r, 0], gt, GT_C, sc)
        axes[r, 1].imshow(img, cmap="gray")
        axes[r, 1].imshow(cam, cmap="jet", alpha=0.38)
        _draw_boxes(axes[r, 1], gt, GT_C, sc, lw=1.2, ls="--")
        _draw_boxes(axes[r, 1], cam_p.get(i, ([], []))[0], CAM_C, sc, lw=2)
        axes[r, 2].imshow(img, cmap="gray")
        _draw_boxes(axes[r, 2], gt, GT_C, sc, lw=1.2, ls="--")
        db, ds = det_p.get(i, ([], []))
        _draw_boxes(axes[r, 2], db, DET_C, sc, lw=2)
        for b, sco in zip(db, ds):
            axes[r, 2].text(b[0] * sc, b[1] * sc - 4, f"{sco:.2f}", color=DET_C, fontsize=7)
        axes[r, 1].set_xlabel(f"p(pneumonia) = {z['test_prob'][k]:.2f}", fontsize=8)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
    axes[0, 0].set_title("X-ray + ground truth", fontsize=9, color=GT_C)
    axes[0, 1].set_title("DenseNet CAM -> boxes (weak)", fontsize=9, color=CAM_C)
    axes[0, 2].set_title("Faster R-CNN (supervised)", fontsize=9, color=DET_C)
    fig.tight_layout()
    fig.savefig(fdir / f"localization_examples_{setting}.png")
    plt.close(fig)


def fig_iou_hist(res, setting, fdir):
    loc = res.get("localization", {}).get(setting, {})
    keys = [k for k in ["DenseNet121+CAM", "FasterRCNN", "DenseNet-gated FasterRCNN"] if k in loc]
    if not keys:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    cols = {"DenseNet121+CAM": CAM_C, "FasterRCNN": DET_C, "DenseNet-gated FasterRCNN": "#0d6b3a"}
    bins = np.linspace(0, 1, 21)
    for k in keys:
        ax.hist(loc[k]["test_iou_values"], bins=bins, histtype="step", lw=1.8, color=cols[k], label=k)
    ax.set_xlabel("IoU with ground truth (per pneumonia test image)")
    ax.set_ylabel("images")
    ax.set_title("IoU distribution (0 = missed or wrong place)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    fig.savefig(fdir / f"iou_hist_{setting}.png")
    plt.close(fig)


def fig_cam_sweep(res, setting, fdir):
    loc = res.get("localization", {}).get(setting, {})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    found = False
    for k, c in [("PaperCNN+CAM", MODEL_COLORS["PaperCNN"]), ("DenseNet121+CAM", MODEL_COLORS["DenseNet121"])]:
        if k in loc:
            sw = loc[k]["val_sweep"]
            t = sorted(sw, key=float)
            ax.plot([float(x) for x in t], [sw[x]["mean_iou"] for x in t], "o-", color=c, label=k)
            found = True
    if not found:
        plt.close(fig)
        return
    ax.set_xlabel("CAM threshold (fraction of map maximum)")
    ax.set_ylabel("validation mean IoU")
    ax.set_title("Tuning the heat-map cut-off", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(fdir / f"cam_threshold_sweep_{setting}.png")
    plt.close(fig)


def make_all_figures(cfg: Config):
    fdir = ensure_dir(Path(cfg.out_dir) / "figures")
    res = load_json(Path(cfg.out_dir) / "results.json")
    df = load_meta(cfg.cache_dir).set_index("idx", drop=False)
    arr = load_cache(cfg.cache_dir, cfg.cache_size)
    fig_data_overview(cfg, df, arr, fdir)
    fig_training(res, fdir)
    fig_paper_comparison(res, fdir)
    for s in cfg.settings:
        fig_roc(res, s, fdir)
        fig_confusion(res, s, "DenseNet121", fdir)
        fig_iou_hist(res, s, fdir)
        fig_cam_sweep(res, s, fdir)
        fig_localization_examples(cfg, df, arr, res, fdir, s)
    print("figures written to", fdir)
