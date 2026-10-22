"""Build notebooks/kaggle_pipeline.ipynb - a self-contained Kaggle notebook.

The notebook embeds the current source files (via %%writefile), so you only
upload ONE file to Kaggle; it always matches the code in this repository.
Re-run this script after changing anything in src/ or scripts/run_all.py.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (sorted((ROOT / "src" / "pneumo").glob("*.py")) + [ROOT / "scripts" / "run_all.py"]
         + [ROOT / "app.py", ROOT / "predict.py"])

RUN = ("!cd /kaggle/working/repo && python scripts/run_all.py "
       "--data-dir $DATA_DIR --cache-dir /tmp/cache --out-dir /kaggle/working/outputs $MODE --stages {}")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


cells = [md("""
# Pneumonia Diagnosis, Detection & Localization - full pipeline (Kaggle)

**UE24CS352A Machine Learning - Mini Project** · Samvith Vaddiparthi (PES1UG24AM468) · Sanjit C (PES1UG24AM448)

Re-implements and extends *Weakly Supervised Pneumonia Localization* (Stanford CS229, 2018):
classical baselines (LR / SVM / RF) → the paper's 10-layer CNN + CAM → ImageNet-pretrained DenseNet-121 + CAM →
supervised Faster R-CNN → classifier-gated detector.

### Before you press *Run All*
1. **Add data**: right panel → *Add Input* → search **RSNA Pneumonia Detection Challenge** (competition) → Add.
   (You must have clicked *Join Competition / I understand and accept* on the competition page once.)
2. **Settings → Accelerator → GPU T4 x1** (or P100).
3. **Settings → Internet → On** (downloads ImageNet / COCO weights once; needs a phone-verified Kaggle account).
4. Choose `MODE` in the next cell: full run ≈ 2-3 h, `--fast` ≈ 45 min.
5. Best: **Save Version → Save & Run All (Commit)** - it keeps running when you close the browser.
   Afterwards download `results_light.zip` and `model_weights.zip` from the *Output* tab.
"""),
         code("""
MODE = ""          # "" = full configuration (recommended);  "--fast" = fewer epochs (~45 min)
import os, glob, subprocess
cands = glob.glob("/kaggle/input/**/stage_2_train_labels.csv", recursive=True)
assert cands, "RSNA dataset not attached - see step 1 above"
DATA_DIR = os.path.dirname(cands[0])
os.environ["DATA_DIR"], os.environ["MODE"] = DATA_DIR, MODE
print("data:", DATA_DIR)
print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True,
                     capture_output=True, text=True).stdout or "NO GPU - enable it in Settings!")
"""),
         code("""
!pip -q install pydicom pylibjpeg pylibjpeg-libjpeg
import urllib.request
try:
    urllib.request.urlopen("https://download.pytorch.org", timeout=10); print("internet OK")
except Exception as e:
    print("WARNING: no internet - pretrained weights cannot be downloaded. Turn Internet ON.", e)
"""),
         md("## 1 · Write the project source code\n(identical to the GitHub repository)")]

for f in FILES:
    rel = f.relative_to(ROOT).as_posix()
    cells.append(code(f"%%writefile /kaggle/working/repo/{rel}\n" + f.read_text()))
cells.insert(len(cells) - len(FILES), code("!mkdir -p /kaggle/working/repo/src/pneumo /kaggle/working/repo/scripts"))

cells += [
    md("## 2 · Prepare data\nCollapse box rows to one row per image, stratified patient-level 70/15/15 split, "
       "decode all 26,684 DICOMs once into a 512×512 uint8 memory-map (~5 min)."),
    code(RUN.format("prepare")),
    md("## 3 · Classical baselines (LR / SVM / RF on 64×64 pixels)"),
    code(RUN.format("baselines")),
    md("## 4 · CNN classifiers: paper-style CNN (128 px) and DenseNet-121 (pretrained)\n"
       "Trained in both settings: **paper** (Normal vs Lung Opacity) and **full** (Lung Opacity vs everything)."),
    code(RUN.format("cnn")),
    md("## 5 · Supervised detector: Faster R-CNN ResNet-50-FPN (COCO-pretrained)"),
    code(RUN.format("detector")),
    md("## 6 · Localization: CAM → boxes, threshold tuning on validation, mean IoU + RSNA mAP on test"),
    code(RUN.format("localize")),
    md("## 7 · Figures and demo samples"),
    code(RUN.format("figures,demo")),
    code("""
import json, pandas as pd
R = json.load(open("/kaggle/working/outputs/results.json"))
for s in R["classification"]:
    print(f"\\n=== classification - {s} setting (test) ===")
    print(pd.DataFrame({k: {m: v[m] for m in ["accuracy", "precision", "recall", "specificity", "f1", "auc"]}
                        for k, v in R["classification"][s].items()}).T.round(4))
for s in R.get("localization", {}):
    print(f"\\n=== localization - {s} setting (test) ===")
    print(pd.DataFrame({k: v["test"] for k, v in R["localization"][s].items()}).T
          [["mean_iou", "rsna_map", "frac_pos_with_box"]].round(4))
print("\\ntimings (min):", R.get("timings_min"))
"""),
    code("""
from IPython.display import Image, display
import glob
for f in sorted(glob.glob("/kaggle/working/outputs/figures/*.png")):
    print(f); display(Image(f, width=900))
"""),
    md("## 8 · Package the outputs\n`results_light.zip` (numbers + figures, small) → upload this to Claude / use for the report.  \n"
       "`model_weights.zip` → put its `models/` folder inside `outputs/` of the repo to run the live demo."),
    code("""
%cd /kaggle/working
!cd outputs && zip -qr ../results_light.zip results.json config.json figures demo_samples
!cd outputs && zip -qr ../model_weights.zip models
!ls -lh /kaggle/working/*.zip
"""),
    md("### Optional: live demo inside Kaggle\nRun the cell below for a public Gradio link (needs Internet on)."),
    code("""
# !pip -q install gradio
# %cd /kaggle/working/repo
# import sys; sys.argv = ["app.py", "--out-dir", "/kaggle/working/outputs", "--share"]
# exec(open("app.py").read())
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = ROOT / "notebooks" / "kaggle_pipeline.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, "with", len(cells), "cells")
