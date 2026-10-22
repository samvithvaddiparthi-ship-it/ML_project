"""Live demo (Gradio).  Upload a chest X-ray (DICOM / PNG / JPG) and get:
  * the probability of pneumonia from DenseNet-121,
  * the Class Activation Map heat-map with weakly-supervised boxes (blue),
  * Faster R-CNN boxes (green), and the ground truth (red) for the bundled
    test examples.

    python app.py                       # http://127.0.0.1:7860
    python app.py --share               # public link (e.g. when running on Kaggle/Colab)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from pneumo.data import read_any_image  # noqa: E402
from pneumo.infer import PneumoniaPredictor, render  # noqa: E402

_cache = {}


def get_predictor(out_dir, setting):
    key = (out_dir, setting)
    if key not in _cache:
        _cache[key] = PneumoniaPredictor(out_dir, setting)
    return _cache[key]


def build(out_dir: str):
    samples_dir = Path(out_dir) / "demo_samples"
    gt = json.loads((samples_dir / "ground_truth.json").read_text()) if (samples_dir / "ground_truth.json").exists() else {}
    examples = sorted(str(p) for p in samples_dir.glob("*.png"))

    def run(file, setting):
        if file is None:
            return None, "Upload an X-ray first."
        path = file if isinstance(file, str) else file.name
        img = read_any_image(path)
        pred = get_predictor(out_dir, setting)(img)
        g = gt.get(Path(path).name, {})
        vis = render(img, pred, g.get("boxes"))
        lines = [f"### p(pneumonia) = **{pred['prob']:.3f}**  "
                 f"(threshold {pred['cls_threshold']:.2f}) → "
                 f"**{'PNEUMONIA' if pred['positive'] else 'no pneumonia'}**",
                 f"- CAM boxes (blue, weakly supervised): {len(pred['cam_boxes'])}",
                 f"- Faster R-CNN boxes (green, supervised): {len(pred['det_boxes'])}"]
        if g:
            lines.append(f"- Ground truth (red): **{g.get('cls', '?')}**, {len(g.get('boxes', []))} box(es)")
        return vis, "\n".join(lines)

    with gr.Blocks(title="Pneumonia detection & localization") as demo:
        gr.Markdown("## Pneumonia Diagnosis, Detection & Localization on Chest X-rays\n"
                    "DenseNet-121 classifier + Class Activation Map (weakly supervised) and "
                    "Faster R-CNN (supervised). Trained on the RSNA Pneumonia Detection Challenge data. "
                    "*Educational project - not a medical device.*")
        with gr.Row():
            with gr.Column(scale=1):
                f = gr.File(label="Chest X-ray (.dcm / .png / .jpg)", file_types=[".dcm", ".png", ".jpg", ".jpeg"],
                            type="filepath")
                setting = gr.Radio(["full", "paper"], value="full", label="Model trained on",
                                   info="full = all RSNA images; paper = Normal vs Pneumonia only")
                btn = gr.Button("Analyse", variant="primary")
                if examples:
                    gr.Examples(examples=[[e] for e in examples], inputs=[f], label="Test-set examples")
            with gr.Column(scale=2):
                out_img = gr.Image(label="Heat-map + boxes", type="numpy")
                out_txt = gr.Markdown()
        btn.click(run, [f, setting], [out_img, out_txt])
        f.change(run, [f, setting], [out_img, out_txt])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build(a.out_dir).launch(share=a.share, server_port=a.port)
