"""Fill the RESULTS section of README.md from outputs/results.json."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"LogReg": "Logistic Regression", "SVM": "SVM (RBF)", "RandomForest": "Random Forest",
         "PaperCNN": "Paper-style CNN", "DenseNet121": "DenseNet-121 (pretrained)"}
SETTING = {"paper": "Normal vs Pneumonia (paper setting)", "full": "Pneumonia vs all (full dataset)"}


def f(x, d=3):
    return "-" if x is None or x != x else f"{x:.{d}f}"


def tables(R: dict) -> str:
    ref = R.get("paper_reference", {})
    ds = R.get("dataset", {})
    s = ["## Results", ""]
    if ds:
        s += [f"Test split of the RSNA data ({sum(ds['split_counts']['test'].values()):,} images; "
              f"total {ds['n_images']:,} images, {ds['n_boxes']:,} boxes). "
              "Every threshold was chosen on the validation split.", ""]
    for setting, cl in R.get("classification", {}).items():
        s += [f"### Classification - {SETTING.get(setting, setting)}", "",
              "| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | Report acc. |",
              "|---|---|---|---|---|---|---|---|"]
        for k in ["LogReg", "SVM", "RandomForest", "PaperCNN", "DenseNet121"]:
            if k in cl:
                m = cl[k]
                r = ref.get("classification_test_accuracy", {}).get(k) if setting == "paper" else None
                s.append(f"| {NAMES[k]} | {f(m['accuracy'])} | {f(m['precision'])} | {f(m['recall'])} | "
                         f"{f(m['specificity'])} | {f(m['f1'])} | {f(m['auc'])} | {f(r) if r else '-'} |")
        s.append("")
    for setting, loc in R.get("localization", {}).items():
        s += [f"### Localization - {SETTING.get(setting, setting)}", "",
              "| Method | Supervision | mean IoU | mean IoU (oracle) | RSNA mAP | pneumonia images with a box |",
              "|---|---|---|---|---|---|"]
        if setting == "paper":
            s += ["| *Report: R-CNN* | boxes | 0.127 | - | - | - |",
                  "| *Report: CNN + CAM* | image labels | 0.151 | 0.379 | - | - |"]
        for k, v in loc.items():
            t = v["test"]
            o = v.get("test_oracle", {}).get("mean_iou")
            s.append(f"| {k} | {v['type']} | {f(t['mean_iou'])} | {f(o) if o is not None else '-'} | "
                     f"{f(t['rsna_map'])} | {t['frac_pos_with_box']*100:.0f}% |")
        s.append("")
    s += ["![comparison](outputs/figures/paper_comparison.png)", "",
          "![examples](outputs/figures/localization_examples_paper.png)", ""]
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "outputs" / "results.json"))
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    readme = ROOT / "README.md"
    txt = readme.read_text()
    new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                 "<!-- RESULTS:START -->\n" + tables(R).replace("\\", "\\\\") + "\n<!-- RESULTS:END -->",
                 txt, flags=re.S)
    readme.write_text(new)
    print("README results section updated")


if __name__ == "__main__":
    main()
