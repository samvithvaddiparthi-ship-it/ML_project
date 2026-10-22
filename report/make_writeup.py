"""Build the 2-page write-up (report/writeup.pdf) from outputs/results.json.

    python report/make_writeup.py [--results outputs/results.json] [--figures outputs/figures]

Every number in the PDF is read from results.json, so the write-up always
matches the run. Needs pdflatex (TeX Live / MiKTeX) or use Overleaf with the
generated writeup.tex + the figures folder.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"LogReg": "Logistic Regression", "SVM": "SVM (RBF)", "RandomForest": "Random Forest",
         "PaperCNN": "Paper-style CNN", "DenseNet121": "DenseNet-121"}


def f(x, d=3):
    return "--" if x is None or x != x else f"{x:.{d}f}"


def pct(x):
    return "--" if x is None or x != x else f"{100 * x:.1f}\\%"


def build_tex(R: dict, figdir: str) -> str:
    ds, cl, loc = R["dataset"], R["classification"], R.get("localization", {})
    cfg = R["config"]
    ref = R["paper_reference"]
    P, F = cl.get("paper", {}), cl.get("full", {})
    LP, LF = loc.get("paper", {}), loc.get("full", {})
    sc = ds["split_counts"]
    n = lambda s: sum(sc[s].values())

    # ---------------- classification table
    rows = []
    for k in ["LogReg", "SVM", "RandomForest", "PaperCNN", "DenseNet121"]:
        if k not in P:
            continue
        a, b = P[k], F.get(k, {})
        r = ref["classification_test_accuracy"].get(k)
        rows.append(f"{NAMES[k]} & {f(r) if r else '--'} & {f(a['accuracy'])} & {f(a['recall'])} & "
                    f"{f(a['auc'])} & {f(b.get('accuracy'))} & {f(b.get('recall'))} & {f(b.get('auc'))} \\\\")
    cls_table = "\n".join(rows)

    # ---------------- localization table
    lrows = ["\\textit{Report: R-CNN} & box & 0.127 & -- & -- & -- \\\\",
             "\\textit{Report: CNN+CAM} & image & 0.151 & 0.379 & -- & -- \\\\ \\midrule"]
    short = {"PaperCNN+CAM": "Paper CNN + CAM", "DenseNet121+CAM": "DenseNet + CAM",
             "FasterRCNN": "Faster R-CNN", "DenseNet-gated FasterRCNN": "Gated Faster R-CNN"}
    sup = {"PaperCNN+CAM": "image", "DenseNet121+CAM": "image", "FasterRCNN": "box",
           "DenseNet-gated FasterRCNN": "box"}
    for k in short:
        if k in LP:
            t, o = LP[k]["test"], LP[k].get("test_oracle", {})
            lrows.append(f"{short[k]} & {sup[k]} & {f(t['mean_iou'])} & {f(o.get('mean_iou'))} & "
                         f"{f(t['rsna_map'])} & {f(LF.get(k, {}).get('test', {}).get('rsna_map'))} \\\\")
    loc_table = "\n".join(lrows)

    d = P.get("DenseNet121", {})
    pc = P.get("PaperCNN", {})
    df_ = F.get("DenseNet121", {})
    cam = LP.get("DenseNet121+CAM", {})
    gated = LP.get("DenseNet-gated FasterRCNN", LP.get("FasterRCNN", {}))
    det = LP.get("FasterRCNN", {})
    t = R.get("timings_min", {})
    train_min = sum(v for k, v in t.items() if k.startswith("train_"))

    best_iou_name, best_iou = max(((short[k], LP[k]["test"]["mean_iou"]) for k in short if k in LP),
                                  key=lambda x: x[1], default=("--", float("nan")))
    best_map_name, best_map = max(((short[k], LP[k]["test"]["rsna_map"]) for k in short if k in LP),
                                  key=lambda x: x[1], default=("--", float("nan")))

    V = {
        "N_IMG": f"{ds['n_images']:,}", "N_BOX": f"{ds['n_boxes']:,}",
        "N_NORM": f"{ds['class_counts'].get('Normal', 0):,}",
        "N_OTHER": f"{ds['class_counts'].get('No Lung Opacity / Not Normal', 0):,}",
        "N_OPAC": f"{ds['class_counts'].get('Lung Opacity', 0):,}",
        "N_TR": f"{n('train'):,}", "N_VA": f"{n('val'):,}", "N_TE": f"{n('test'):,}",
        "POS_PAPER": pct(ds.get("positive_rate_paper")), "POS_FULL": pct(ds.get("positive_rate_full")),
        "BOX_W": f"{ds.get('box_w_mean', 0):.0f}", "BOX_H": f"{ds.get('box_h_mean', 0):.0f}",
        "BOX_AREA": pct(ds.get("box_area_frac_mean")),
        "MULTI": pct(1 - ds["boxes_per_positive"].get("1", 0) / max(sum(ds["boxes_per_positive"].values()), 1)),
        "CLS_TABLE": cls_table, "LOC_TABLE": loc_table,
        "D_ACC": f(d.get("accuracy")), "D_AUC": f(d.get("auc")), "D_REC": f(d.get("recall")),
        "D_SPEC": f(d.get("specificity")), "D_THR": f(d.get("threshold"), 2),
        "PC_ACC": f(pc.get("accuracy")), "PC_AUC": f(pc.get("auc")),
        "DF_ACC": f(df_.get("accuracy")), "DF_AUC": f(df_.get("auc")), "DF_REC": f(df_.get("recall")),
        "CAM_IOU": f(cam.get("test", {}).get("mean_iou")), "CAM_ORC": f(cam.get("test_oracle", {}).get("mean_iou")),
        "CAM_THR": f(cam.get("cam_threshold"), 1),
        "DET_IOU": f(det.get("test", {}).get("mean_iou")), "DET_MAP": f(det.get("test", {}).get("rsna_map")),
        "G_MAP": f(gated.get("test", {}).get("rsna_map")), "G_IOU": f(gated.get("test", {}).get("mean_iou")),
        "G_GATE": f(gated.get("gate"), 1), "G_SC": f(gated.get("score_threshold"), 1),
        "BEST_IOU": f(best_iou), "BEST_IOU_NAME": best_iou_name,
        "BEST_MAP": f(best_map), "BEST_MAP_NAME": best_map_name,
        "IOU_GAIN": f"{best_iou / 0.1508:.1f}" if best_iou == best_iou else "--",
        "DN_SIZE": str(cfg["densenet_size"]), "DN_EP": str(cfg["densenet_epochs"]),
        "PC_EP": str(cfg["paper_cnn_epochs"]), "DET_EP": str(cfg["det_epochs"]), "DET_SIZE": str(cfg["det_size"]),
        "TRAIN_MIN": f"{train_min:.0f}", "TOTAL_MIN": f"{sum(v for k, v in t.items() if k.startswith('stage_')):.0f}",
        "FIG": figdir,
    }
    tex = TEMPLATE
    for k, v in V.items():
        tex = tex.replace(f"<<{k}>>", v)
    return tex


TEMPLATE = r"""
\documentclass[10pt,twocolumn]{article}
\usepackage[a4paper,top=1.3cm,bottom=1.4cm,left=1.4cm,right=1.4cm,columnsep=0.55cm]{geometry}
\usepackage{mathptmx}
\usepackage{booktabs,graphicx,xcolor,enumitem,titlesec,microtype}
\usepackage[font=footnotesize,labelfont=bf,skip=3pt]{caption}
\usepackage[hidelinks]{hyperref}
\titleformat{\section}{\normalsize\bfseries\color[HTML]{1F3A68}}{\thesection}{0.5em}{}
\titlespacing*{\section}{0pt}{5pt}{2pt}
\setlist{nosep,leftmargin=1.1em}
\setlength{\parskip}{1.5pt}\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.05}
\begin{document}
\twocolumn[{\begin{center}
{\Large\bfseries Pneumonia Diagnosis, Detection and Localization on Chest X-rays}\\[3pt]
{\small Samvith Vaddiparthi (PES1UG24AM468) \quad Sanjit C (PES1UG24AM448)}\\
{\footnotesize UE24CS352A Machine Learning -- Mini Project, PES University \quad
Reference: Huang, Cortes \& Monam, \emph{Weakly Supervised Pneumonia Localization}, Stanford CS229 (2018)}
\end{center}\vspace{2pt}}]

\section{Problem statement}
Pneumonia causes over 15\% of deaths of children under five, and chest X-rays are the main diagnostic tool, yet
hospital images are usually labelled only ``pneumonia / no pneumonia'' without marking \emph{where} the infection is.
Given a frontal chest X-ray we (i) \textbf{diagnose} pneumonia (binary classification) and (ii) \textbf{localize}
the lung opacities with bounding boxes. Following the reference report we study \emph{weakly supervised}
localization, which needs only image labels (a classifier plus Class Activation Maps), and compare it with a
\emph{supervised} detector trained on boxes. Our goals were to reproduce the report, then improve on its two
headline numbers: 92.5\% test accuracy and 0.151 mean IoU.

\section{Dataset}
RSNA Pneumonia Detection Challenge (Kaggle, stage-2 labels): <<N_IMG>> chest X-rays (1024$\times$1024 DICOM),
<<N_OPAC>> \emph{Lung Opacity} (pneumonia), <<N_NORM>> \emph{Normal}, and <<N_OTHER>> \emph{No Lung Opacity / Not Normal}
(other diseases that also make the lungs look cloudy). The <<N_BOX>> ground-truth boxes are on average
<<BOX_W>>$\times$<<BOX_H>> px (<<BOX_AREA>> of the image); <<MULTI>> of pneumonia images have more than one box.
We split \emph{by patient}, stratified by class: 70/15/15 = <<N_TR>> / <<N_VA>> / <<N_TE>> images. Thresholds
are tuned only on validation; every number below is on the untouched test split. Two settings:
\textbf{paper} = Normal vs Lung Opacity (as in the report, <<POS_PAPER>> positive) and \textbf{full} =
Lung Opacity vs everything (<<POS_FULL>> positive), the realistic and harder task.

\section{Approach}
\begin{itemize}
\item \textbf{Baselines:} Logistic Regression, RBF-SVM (on 150 PCA components) and Random Forest on 64$\times$64 flattened pixels.
\item \textbf{Paper-style CNN (replication):} 10 conv layers (3$\times$3, BatchNorm, ReLU, 3 max-pools)
$\rightarrow$ global average pooling (GAP) $\rightarrow$ one FC layer; 128\,px input, Adam $10^{-4}$, <<PC_EP>> epochs.
\item \textbf{DenseNet-121 (improvement):} the CheXNet backbone, ImageNet-pretrained, GAP + Linear head, <<DN_SIZE>>\,px,
affine/intensity/flip augmentation, AdamW + cosine schedule, mixed precision, <<DN_EP>> epochs; class imbalance via
\texttt{pos\_weight}; best epoch by validation AUC; decision threshold maximising validation F1.
\item \textbf{Weak localization (CAM):} $M(x,y)=\sum_k w_k f_k(x,y)$ from the last feature map and FC weights $\rightarrow$
up-sample $\rightarrow$ normalise by the maximum $\rightarrow$ threshold $\rightarrow$ connected components (the report's
DFS clusters) $\rightarrow$ one box per blob. The cut-off is tuned on validation (the report used a fixed value and
noted it should adapt). Only images the classifier calls positive receive boxes.
\item \textbf{Supervised localization:} Faster R-CNN ResNet-50-FPN, COCO-pretrained, <<DET_SIZE>>\,px, <<DET_EP>> epochs on all
pneumonia images plus an equal fresh sample of negatives each epoch. \textbf{Gating:} detector boxes are kept only
when DenseNet's $p(\mathrm{pneumonia})$ exceeds a validation-tuned gate.
\item \textbf{Metrics:} accuracy, precision, recall, specificity, F1, ROC-AUC; the report's \textbf{mean IoU} (IoU of
predicted vs.\ ground-truth box unions over pneumonia images, 0 when nothing is predicted; \emph{oracle} = every
pneumonia image localized), and the official \textbf{RSNA mAP} (mean of TP/(TP+FP+FN) over IoU 0.40--0.75, which
also penalises boxes on healthy lungs).
\end{itemize}

\section{Implementation}
Python 3, PyTorch/torchvision, scikit-learn, pydicom, OpenCV; trained on one Kaggle T4 GPU (<<TRAIN_MIN>> min of
training, <<TOTAL_MIN>> min end to end). The pipeline is split into resumable stages (\texttt{prepare},
\texttt{baselines}, \texttt{cnn}, \texttt{detector}, \texttt{localize}, \texttt{figures}); every DICOM is decoded once into a
512$\times$512 memory-mapped cache. All results go to one \texttt{results.json}, from which the figures, README tables and
this write-up are generated. The repository also has unit tests for the metrics, a CPU smoke test on synthetic
RSNA-format data, a one-click Kaggle notebook, a CLI (\texttt{predict.py}) and a Gradio live demo (\texttt{app.py}) that
shows the probability, CAM heat-map and boxes for an uploaded X-ray.

\begin{table}[t]
\centering\footnotesize
\caption{Classification on the test split. Accuracy/recall at the validation-chosen threshold.}
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
& \multicolumn{4}{c}{\textbf{paper setting}} & \multicolumn{3}{c}{\textbf{full setting}}\\
\cmidrule(lr){2-5}\cmidrule(l){6-8}
Model & Report & Acc. & Recall & AUC & Acc. & Recall & AUC\\
\midrule
<<CLS_TABLE>>
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering\footnotesize
\caption{Localization on the test split (paper setting unless marked).}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llcccc@{}}
\toprule
Method & Labels & mIoU & mIoU orc. & mAP & mAP (full)\\
\midrule
<<LOC_TABLE>>
\bottomrule
\end{tabular}
\end{table}

\section{Results}
\textbf{Diagnosis.} In the paper setting the pretrained DenseNet-121 reaches <<D_ACC>> accuracy and <<D_AUC>> ROC-AUC
(recall <<D_REC>>, specificity <<D_SPEC>>), against <<PC_ACC>> for our from-scratch replication of the report's CNN
and 0.925 reported. Pixel-based baselines are far behind, as in the report. On the full dataset accuracy
drops to <<DF_ACC>> (AUC <<DF_AUC>>), because ``not normal'' lungs also show opacities.

\textbf{Localization.} CAM boxes from DenseNet (cut-off <<CAM_THR>> of the map maximum) give a mean IoU of <<CAM_IOU>>
(<<CAM_ORC>> if every pneumonia image is localized), with no box labels used. The supervised Faster R-CNN reaches
<<DET_IOU>> mean IoU and <<DET_MAP>> RSNA mAP; gating it with the classifier (gate <<G_GATE>>, score $\geq$ <<G_SC>>)
gives <<G_MAP>> mAP. The best mean IoU, <<BEST_IOU>> (<<BEST_IOU_NAME>>), is <<IOU_GAIN>>$\times$ the report's 0.151.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth,trim=0 0 0 0,clip]{<<FIG>>/paper_comparison.png}
\caption{Our models vs.\ the reference report (paper setting, test split).}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.92\linewidth]{<<FIG>>/localization_examples_strip.png}
\caption{Test X-rays. Red: ground truth; blue: DenseNet CAM boxes (weakly supervised); green: Faster R-CNN.}
\end{figure}

\section{Conclusions}
\begin{itemize}
\item Transfer learning matters most: an ImageNet-pretrained DenseNet at higher resolution beats the report's
from-scratch CNN on diagnosis, and its CAMs localize better.
\item Weak supervision gives useful heat-maps without any box labels, but CAM boxes are coarse; a detector trained on
boxes localizes more tightly, and gating it with the classifier removes most false alarms on healthy lungs.
\item The full 3-class setting is clearly harder than the report's 2-class setting, so the report's numbers
overstate real-world performance.
\item Limitations / future work: 1024\,px inputs, lung segmentation, test-time augmentation, ensembling,
Grad-CAM++ and a radiologist-checked error analysis. Educational project, not a medical device.
\end{itemize}

{\footnotesize\textbf{References.} [1] Huang et al., Weakly Supervised Pneumonia Localization, CS229 2018.
[2] Zhou et al., CAM, CVPR 2016. [3] Rajpurkar et al., CheXNet, 2017. [4] Huang et al., DenseNet, CVPR 2017.
[5] Ren et al., Faster R-CNN, NeurIPS 2015. [6] RSNA Pneumonia Detection Challenge, Kaggle 2018.\par}
\end{document}
"""


def make_strip(figdir: Path):
    """Crop the first 3 rows of localization_examples_paper.png into a compact strip."""
    from PIL import Image
    src = figdir / "localization_examples_paper.png"
    if not src.exists():
        return
    im = Image.open(src)
    w, h = im.size
    rows = 6
    im.crop((0, 0, w, int(h * 3 / rows))).save(figdir / "localization_examples_strip.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "outputs" / "results.json"))
    ap.add_argument("--figures", default=str(ROOT / "outputs" / "figures"))
    ap.add_argument("--out", default=str(ROOT / "report" / "writeup.pdf"))
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    figdir = Path(a.figures).resolve()
    make_strip(figdir)
    tex = build_tex(R, figdir.as_posix())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    texf = out.with_suffix(".tex")
    texf.write_text(tex)
    if shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", str(out.parent), str(texf)],
                           stdout=subprocess.DEVNULL, check=False)
        print("wrote", out)
    else:
        print("pdflatex not found - compile", texf, "on Overleaf")


if __name__ == "__main__":
    main()
