"""Create a tiny fake dataset in the exact RSNA folder/CSV format.

Used only for the smoke test (`python scripts/run_all.py --smoke`) so that the
whole pipeline can be checked on a laptop CPU in a few minutes without the
real 3.5 GB Kaggle download.  'Pneumonia' images get bright blurry blobs inside
their boxes; 'not normal' images get sharp-edged blobs; normal images get none.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def write_dicom(path: Path, img: np.ndarray, pid: str) -> None:
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.PatientID, ds.Modality = pid, "CR"
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = img.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
    ds.PixelData = img.astype(np.uint8).tobytes()
    try:
        ds.save_as(str(path), enforce_file_format=True)     # pydicom >= 3
    except TypeError:
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.save_as(str(path), write_like_original=False)     # pydicom 2.x


def chest(rng, size):
    """A crude chest-X-ray-like background: dark lungs inside a brighter body."""
    img = np.full((size, size), 40, np.float32)
    cv2.ellipse(img, (size // 2, size // 2), (int(size * .45), int(size * .48)), 0, 0, 360, 150, -1)
    for cx in (0.32, 0.68):
        cv2.ellipse(img, (int(size * cx), int(size * .5)), (int(size * .14), int(size * .3)), 0, 0, 360, 55, -1)
    img += rng.normal(0, 12, img.shape)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_rsna")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--size", type=int, default=256, help="DICOM size (real data is 1024)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(a.out)
    (out / "stage_2_train_images").mkdir(parents=True, exist_ok=True)
    rows, info = [], []
    for k in range(a.n):
        pid = f"synthetic-{k:05d}"
        cls = rng.choice(["Normal", "No Lung Opacity / Not Normal", "Lung Opacity"], p=[.35, .35, .30])
        img = chest(rng, a.size)
        if cls == "Lung Opacity":
            for _ in range(rng.integers(1, 3)):
                cx = rng.choice([0.32, 0.68]) * a.size + rng.normal(0, a.size * .03)
                cy = rng.uniform(.35, .65) * a.size
                w, h = rng.uniform(.12, .2) * a.size, rng.uniform(.15, .3) * a.size
                blob = np.zeros_like(img)
                cv2.ellipse(blob, (int(cx), int(cy)), (int(w / 2), int(h / 2)), 0, 0, 360, 70, -1)
                img += cv2.GaussianBlur(blob, (0, 0), a.size * .02)
                rows.append([pid, cx - w / 2, cy - h / 2, w, h, 1])
        else:
            if cls != "Normal":
                cv2.circle(img, (int(rng.uniform(.25, .75) * a.size), int(rng.uniform(.3, .7) * a.size)),
                           int(a.size * .04), 140, -1)
            rows.append([pid, np.nan, np.nan, np.nan, np.nan, 0])
        info.append([pid, cls])
        write_dicom(out / "stage_2_train_images" / f"{pid}.dcm", np.clip(img, 0, 255), pid)
    pd.DataFrame(rows, columns=["patientId", "x", "y", "width", "height", "Target"]).to_csv(
        out / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(info, columns=["patientId", "class"]).to_csv(out / "stage_2_detailed_class_info.csv", index=False)
    print(f"wrote {a.n} synthetic DICOMs to {out}")


if __name__ == "__main__":
    main()
