"""Classical ML baselines on flattened pixels (as in the Stanford report).

LR and SVM work on a PCA projection of the standardised pixels (4096 raw
features is more than the classical models can use well); Random Forest uses
the raw pixels directly.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def flatten_images(arr, idx, size: int) -> np.ndarray:
    out = np.empty((len(idx), size * size), dtype=np.float32)
    for j, i in enumerate(idx):
        out[j] = cv2.resize(np.asarray(arr[i]), (size, size), interpolation=cv2.INTER_AREA).ravel() / 255.0
    return out


def build_baselines(n_pca: int, seed: int = 42) -> dict:
    return {
        "LogReg": make_pipeline(StandardScaler(), PCA(n_pca, random_state=seed),
                                LogisticRegression(max_iter=2000, C=1.0)),
        "SVM": make_pipeline(StandardScaler(), PCA(n_pca, random_state=seed),
                             SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                               random_state=seed),
    }


def scores(model, X) -> np.ndarray:
    """Probability-like score in [0, 1] (SVM margin squashed with a sigmoid)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return 1 / (1 + np.exp(-d))
