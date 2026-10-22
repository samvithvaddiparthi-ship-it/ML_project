"""Data loading for the RSNA Pneumonia Detection Challenge.

Files expected in `data_dir` (as provided by Kaggle):
    stage_2_train_labels.csv        patientId, x, y, width, height, Target
    stage_2_detailed_class_info.csv patientId, class
    stage_2_train_images/<patientId>.dcm

One row of `stage_2_train_labels.csv` is one bounding box, so a patient with
two opacities has two rows. We collapse these into one row per image with a
list of boxes, split *by patient* (one image per patient, so no leakage), and
decode every DICOM exactly once into a uint8 memory-mapped array.
"""
from __future__ import annotations

import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

CLASS_NORMAL = "Normal"
CLASS_OPACITY = "Lung Opacity"
CLASS_OTHER = "No Lung Opacity / Not Normal"
CLASSES = [CLASS_NORMAL, CLASS_OTHER, CLASS_OPACITY]


# ----------------------------------------------------------------------------
# label table
# ----------------------------------------------------------------------------
def build_label_table(data_dir: str | Path) -> pd.DataFrame:
    """One row per image: patientId, target, cls, boxes (list of [x, y, w, h])."""
    data_dir = Path(data_dir)
    labels = pd.read_csv(data_dir / "stage_2_train_labels.csv")
    info = pd.read_csv(data_dir / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")

    def _boxes(g: pd.DataFrame) -> list:
        g = g.dropna(subset=["x"])
        return g[["x", "y", "width", "height"]].astype(float).values.tolist()

    rows = []
    for pid, g in labels.groupby("patientId", sort=False):
        rows.append({"patientId": pid, "target": int(g["Target"].max()), "boxes": _boxes(g)})
    df = pd.DataFrame(rows).merge(info, on="patientId", how="left")
    df = df.rename(columns={"class": "cls"})
    df["n_boxes"] = df["boxes"].apply(len)
    return df.reset_index(drop=True)


def make_split(df: pd.DataFrame, fracs=(0.7, 0.15, 0.15), seed: int = 42) -> pd.DataFrame:
    """Stratified (by 3-way class) patient-level train/val/test split."""
    tr, va, te = fracs
    idx = np.arange(len(df))
    i_tr, i_rest = train_test_split(idx, train_size=tr, stratify=df["cls"], random_state=seed)
    rel = va / (va + te)
    i_va, i_te = train_test_split(i_rest, train_size=rel, stratify=df["cls"].iloc[i_rest],
                                  random_state=seed)
    split = np.empty(len(df), dtype=object)
    split[i_tr], split[i_va], split[i_te] = "train", "val", "test"
    df = df.copy()
    df["split"] = split
    return df


def setting_subset(df: pd.DataFrame, setting: str) -> pd.DataFrame:
    """`paper`: Normal vs Lung Opacity only (the Stanford report removed the
    'No Lung Opacity / Not Normal' images).  `full`: all images; positive =
    Lung Opacity, negative = everything else (the real, harder task)."""
    if setting == "paper":
        return df[df["cls"] != CLASS_OTHER]
    if setting == "full":
        return df
    raise ValueError(setting)


# ----------------------------------------------------------------------------
# DICOM decoding + caching
# ----------------------------------------------------------------------------
def read_dicom(path: str | Path) -> np.ndarray:
    """Return a uint8 2-D array (0 = black) from a DICOM file."""
    import pydicom
    ds = pydicom.dcmread(str(path))
    img = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        img = img.max() - img
    lo, hi = float(img.min()), float(img.max())
    img = (img - lo) / max(hi - lo, 1e-6) * 255.0
    return img.astype(np.uint8)


def read_any_image(path: str | Path) -> np.ndarray:
    """DICOM / PNG / JPG -> uint8 grayscale (used by the demo and predict.py)."""
    path = str(path)
    if path.lower().endswith(".dcm"):
        return read_dicom(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img


def _load_resize(pid: str, img_dir: str, size: int):
    img = read_dicom(Path(img_dir) / f"{pid}.dcm")
    h, w = img.shape
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), h, w


def build_cache(df: pd.DataFrame, data_dir: str | Path, cache_dir: str | Path,
                size: int = 512, workers: int = 4) -> pd.DataFrame:
    """Decode all DICOMs once into `cache_dir/img{size}.npy` (memmap, N x size x size).

    Adds columns `idx` (row in the cache), `orig_h`, `orig_w` and writes
    `cache_dir/meta.csv`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_dir = Path(data_dir) / "stage_2_train_images"
    arr = np.lib.format.open_memmap(cache_dir / f"img{size}.npy", mode="w+",
                                    dtype=np.uint8, shape=(len(df), size, size))
    hs, ws = np.zeros(len(df), int), np.zeros(len(df), int)
    fn = partial(_load_resize, img_dir=str(img_dir), size=size)
    pids = df["patientId"].tolist()
    if workers > 1:
        with Pool(workers) as pool:
            it = pool.imap(fn, pids, chunksize=32)
            for i, (img, h, w) in enumerate(it):
                arr[i], hs[i], ws[i] = img, h, w
                if i % 2000 == 0:
                    print(f"  cached {i}/{len(df)}", flush=True)
    else:
        for i, pid in enumerate(pids):
            arr[i], hs[i], ws[i] = fn(pid)
    arr.flush()
    df = df.copy()
    df["idx"], df["orig_h"], df["orig_w"] = np.arange(len(df)), hs, ws
    out = df.copy()
    out["boxes"] = out["boxes"].apply(json.dumps)
    out.to_csv(cache_dir / "meta.csv", index=False)
    return df


def load_meta(cache_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(cache_dir) / "meta.csv")
    df["boxes"] = df["boxes"].apply(json.loads)
    return df


def load_cache(cache_dir: str | Path, size: int) -> np.ndarray:
    return np.load(Path(cache_dir) / f"img{size}.npy", mmap_mode="r")


# ----------------------------------------------------------------------------
# augmentation (numpy / OpenCV, so it works in DataLoader workers)
# ----------------------------------------------------------------------------
def random_affine(img: np.ndarray, rng: np.random.Generator, rot=10, scale=(0.9, 1.1),
                  shift=0.06) -> np.ndarray:
    h, w = img.shape
    ang = rng.uniform(-rot, rot)
    sc = rng.uniform(*scale)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    m[:, 2] += rng.uniform(-shift, shift, 2) * np.array([w, h])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def random_intensity(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = rng.uniform(0.9, 1.1)          # contrast
    b = rng.uniform(-12, 12)           # brightness
    return np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(img: np.ndarray, three_channel: bool):
    """uint8 HxW -> float tensor. 1-ch: (x-mean)/std per image. 3-ch: ImageNet norm."""
    import torch
    x = img.astype(np.float32) / 255.0
    if three_channel:
        x = np.repeat(x[None], 3, 0)
        x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    else:
        x = ((x - x.mean()) / (x.std() + 1e-6))[None]
    return torch.from_numpy(np.ascontiguousarray(x))


# ----------------------------------------------------------------------------
# torch datasets
# ----------------------------------------------------------------------------
class ClsDataset:
    """Image-level classification dataset (pneumonia vs not)."""

    def __init__(self, df: pd.DataFrame, cache_dir, cache_size: int, size: int,
                 three_channel: bool, augment: bool, seed: int = 0):
        self.idx = df["idx"].values
        self.y = df["target"].values.astype(np.float32)
        self.cache_dir, self.cache_size = cache_dir, cache_size
        self.size, self.three, self.augment = size, three_channel, augment
        self._arr = None
        self.seed = seed

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        import torch
        if self._arr is None:
            self._arr = load_cache(self.cache_dir, self.cache_size)
        img = np.asarray(self._arr[self.idx[i]])
        if self.size != self.cache_size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.augment:
            rng = np.random.default_rng()
            img = random_affine(img, rng)
            img = random_intensity(img, rng)
            if rng.random() < 0.5:
                img = img[:, ::-1]
        return to_tensor(img, self.three), torch.tensor(self.y[i])


class DetDataset:
    """Detection dataset for torchvision Faster R-CNN (boxes in cache pixels)."""

    def __init__(self, df: pd.DataFrame, cache_dir, cache_size: int, size: int, augment: bool):
        self.df = df.reset_index(drop=True)
        self.cache_dir, self.cache_size, self.size, self.augment = cache_dir, cache_size, size, augment
        self._arr = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import torch
        if self._arr is None:
            self._arr = load_cache(self.cache_dir, self.cache_size)
        r = self.df.iloc[i]
        img = np.asarray(self._arr[r["idx"]])
        if self.size != self.cache_size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        sx, sy = self.size / r["orig_w"], self.size / r["orig_h"]
        boxes = np.array([[x * sx, y * sy, (x + w) * sx, (y + h) * sy] for x, y, w, h in r["boxes"]],
                         dtype=np.float32).reshape(-1, 4)
        if self.augment:
            rng = np.random.default_rng()
            img = random_intensity(img, rng)
            if rng.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                if len(boxes):
                    boxes[:, [0, 2]] = self.size - boxes[:, [2, 0]]
        x = torch.from_numpy(np.repeat((img.astype(np.float32) / 255.0)[None], 3, 0))
        target = {"boxes": torch.from_numpy(boxes),
                  "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return x, target, int(i)


def det_collate(batch):
    return tuple(zip(*batch))
