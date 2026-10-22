"""Small helpers shared by all stages."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=_to_jsonable))


def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text())


def update_results(out_dir: str | Path, keys: list[str], value) -> None:
    """Write `value` into outputs/results.json at the nested location `keys`.

    Every stage writes its own part, so stages can be re-run independently.
    """
    path = Path(out_dir) / "results.json"
    res = load_json(path)
    d = res
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    save_json(res, path)


@contextmanager
def timer(name: str, out_dir: str | Path | None = None):
    t0 = time.time()
    print(f"[{name}] started", flush=True)
    yield
    dt = time.time() - t0
    print(f"[{name}] finished in {dt/60:.1f} min", flush=True)
    if out_dir is not None:
        update_results(out_dir, ["timings_min", name], round(dt / 60, 2))
