"""Training / inference loops for the CNN classifiers and the detector."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .data import det_collate


def _amp(device):
    return device.type == "cuda"


@torch.no_grad()
def predict_probs(model, ds, device, batch: int = 64, workers: int = 2) -> np.ndarray:
    model.eval()
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    out = []
    for x, _ in dl:
        with torch.autocast(device.type, enabled=_amp(device)):
            out.append(torch.sigmoid(model(x.to(device, non_blocking=True)).float()).cpu())
    return torch.cat(out).numpy()


def train_classifier(model, train_ds, val_ds, device, epochs: int, lr: float, batch: int,
                     workers: int = 4, weight_decay: float = 1e-4, log=print):
    """Adam(W) + cosine schedule + BCE (pos_weight balances classes).
    Keeps the weights of the epoch with the best validation ROC-AUC."""
    model.to(device)
    y = train_ds.y
    pos_weight = torch.tensor([(1 - y.mean()) / max(y.mean(), 1e-6)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    dl = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                    drop_last=len(train_ds) > batch, pin_memory=device.type == "cuda",
                    persistent_workers=workers > 0)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, len(dl) // 2)) * 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler(enabled=_amp(device))
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": [], "val_acc": []}
    best_key, best_state = None, None
    for ep in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, t in dl:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=_amp(device)):
                loss = crit(model(x).float(), t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        pv = predict_probs(model, val_ds, device, batch=batch * 2, workers=workers)
        yv = val_ds.y
        eps = 1e-6
        vloss = float(-np.mean(yv * np.log(pv + eps) + (1 - yv) * np.log(1 - pv + eps)))
        auc = roc_auc_score(yv, pv) if len(np.unique(yv)) > 1 else float("nan")
        acc = float(((pv >= 0.5) == yv).mean())
        for k, v in zip(hist, [ep, float(np.mean(losses)), vloss, float(auc), acc]):
            hist[k].append(v)
        log(f"  epoch {ep}/{epochs}  train_loss {np.mean(losses):.4f}  val_loss {vloss:.4f}  "
            f"val_auc {auc:.4f}  val_acc {acc:.4f}  ({time.time()-t0:.0f}s)")
        # select by validation AUC, ties broken by lower validation loss
        key = (round(float(np.nan_to_num(auc, nan=0.0)), 4), -vloss)
        if best_key is None or key > best_key:
            best_key, best_state = key, copy.deepcopy(model.state_dict())
            hist["best_epoch"] = ep
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


# ------------------------------------------------------------------ detector
def train_detector(model, ds_pos, ds_neg, device, epochs: int, lr: float, batch: int,
                   neg_ratio: float = 1.0, workers: int = 4, log=print):
    """Faster R-CNN fine-tuning. Each epoch uses every positive image plus a
    fresh random sample of `neg_ratio * n_pos` negatives (empty targets teach
    the detector to stay silent on healthy lungs)."""
    from torch.utils.data import ConcatDataset, Subset
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    n_neg = min(len(ds_neg), int(neg_ratio * len(ds_pos)))
    steps_per_epoch = math.ceil((len(ds_pos) + n_neg) / batch)
    total = epochs * steps_per_epoch
    warm = min(300, steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler(enabled=_amp(device))
    hist = {"epoch": [], "train_loss": []}
    rng = np.random.default_rng(0)
    for ep in range(1, epochs + 1):
        model.train()
        neg_idx = rng.choice(len(ds_neg), n_neg, replace=False) if n_neg else []
        ds = ConcatDataset([ds_pos, Subset(ds_neg, list(neg_idx))]) if n_neg else ds_pos
        dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                        collate_fn=det_collate, pin_memory=device.type == "cuda")
        t0, losses = time.time(), []
        for imgs, targets, _ in dl:
            imgs = [i.to(device, non_blocking=True) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=_amp(device)):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        hist["epoch"].append(ep)
        hist["train_loss"].append(float(np.mean(losses)))
        log(f"  det epoch {ep}/{epochs}  loss {np.mean(losses):.4f}  ({time.time()-t0:.0f}s)")
    return model, hist


@torch.no_grad()
def predict_detector(model, ds, device, batch: int = 8, workers: int = 4):
    """Returns per image (boxes [N,4] in *dataset pixels*, scores [N])."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=det_collate)
    out = [None] * len(ds)
    for imgs, _, ids in dl:
        imgs = [i.to(device) for i in imgs]
        with torch.autocast(device.type, enabled=_amp(device)):
            res = model(imgs)
        for r, i in zip(res, ids):
            out[i] = (r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy())
    return out
