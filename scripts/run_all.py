"""Run the whole experiment (or selected stages).

Examples
--------
Full run on Kaggle (GPU):
    python scripts/run_all.py --data-dir /kaggle/input/rsna-pneumonia-detection-challenge \
        --cache-dir /tmp/cache --out-dir /kaggle/working/outputs

Quicker run (~45 min, fewer epochs):
    python scripts/run_all.py --fast ...

Only re-do localization + figures after training:
    python scripts/run_all.py --stages localize,figures ...

CPU smoke test on synthetic data (no download needed):
    python scripts/run_all.py --smoke
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pneumo.config import Config  # noqa: E402
from pneumo.pipeline import run  # noqa: E402

ALL = ["prepare", "baselines", "cnn", "detector", "localize", "figures", "demo"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--fast", action="store_true", help="fewer epochs (~45 min on a T4)")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic CPU run")
    ap.add_argument("--limit", type=int, help="use only N images")
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()

    if a.smoke:
        syn = ROOT / "synthetic_rsna"
        if not (syn / "stage_2_train_labels.csv").exists():
            subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"),
                                   "--out", str(syn)])
        cfg = Config.smoke(data_dir=str(syn), cache_dir=str(ROOT / "cache_smoke"),
                           out_dir=str(ROOT / "outputs_smoke"))
    else:
        cfg = Config.fast() if a.fast else Config()
    for k, v in [("data_dir", a.data_dir), ("cache_dir", a.cache_dir), ("out_dir", a.out_dir),
                 ("limit", a.limit), ("num_workers", a.workers)]:
        if v is not None:
            setattr(cfg, k, v)
    if a.no_detector:
        cfg.run_detector = False
    if a.no_pretrained:
        cfg.pretrained = False
    run(cfg, [s.strip() for s in a.stages.split(",") if s.strip()])


if __name__ == "__main__":
    main()
