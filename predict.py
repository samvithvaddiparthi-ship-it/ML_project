"""Command-line prediction on one or more chest X-rays (DICOM / PNG / JPG).

    python predict.py path/to/xray.dcm [more files] --out-dir outputs --save-dir predictions
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pneumo.data import read_any_image  # noqa: E402
from pneumo.infer import PneumoniaPredictor, render  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out-dir", default="outputs", help="folder with results.json and models/")
    ap.add_argument("--setting", default="full", choices=["full", "paper"])
    ap.add_argument("--save-dir", default="predictions")
    a = ap.parse_args()
    pred = PneumoniaPredictor(a.out_dir, a.setting)
    Path(a.save_dir).mkdir(exist_ok=True)
    for p in a.images:
        img = read_any_image(p)
        r = pred(img)
        verdict = "PNEUMONIA" if r["positive"] else "no pneumonia"
        print(f"{p}: p(pneumonia)={r['prob']:.3f} -> {verdict}; "
              f"CAM boxes {[[round(v) for v in b] for b in r['cam_boxes']]}; "
              f"detector boxes {[[round(v) for v in b] for b in r['det_boxes']]}")
        out = Path(a.save_dir) / (Path(p).stem + "_pred.png")
        cv2.imwrite(str(out), render(img, r)[:, :, ::-1])
    print("visualisations saved to", a.save_dir)


if __name__ == "__main__":
    main()
