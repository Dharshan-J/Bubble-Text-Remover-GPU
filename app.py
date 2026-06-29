#!/usr/bin/env python3
"""Web UI for manga-bubble-remover (Gradio).

    pip install gradio        # or: pip install -r requirements-cu121.txt
    python app.py             # then open the printed http://127.0.0.1:7860

Drag in one or many pages, tweak the sliders, hit Run. Cleaned pages show in the
gallery and a ZIP of all results is offered for download.
"""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import torch

from mbr import io_utils
from mbr.detector import TextDetector
from mbr.inpainter import LamaInpainter
from mbr.pipeline import Config, process_one

MODELS_DIR = Path("models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazily-loaded singletons so the models load once, on first Run.
_STATE: dict = {"detector": None, "inpainter": None}


def _ensure_models():
    if _STATE["detector"] is None:
        det = MODELS_DIR / "comictextdetector.pt.onnx"
        lama = MODELS_DIR / "big-lama.pt"
        if not det.exists() or not lama.exists():
            raise gr.Error(
                "Model files missing. Run `python download_models.py` first."
            )
        _STATE["detector"] = TextDetector(str(det), device=DEVICE)
        _STATE["inpainter"] = LamaInpainter(str(lama), device=DEVICE)
    return _STATE["detector"], _STATE["inpainter"]


def process(files, mask_source, threshold, dilation, closing, crop_pad, full_image, progress=gr.Progress()):
    if not files:
        raise gr.Error("Add at least one image.")
    detector, inpainter = _ensure_models()

    cfg = Config(
        detector_model="", lama_model="", device=DEVICE,
        mask_source=mask_source, mask_threshold=int(threshold),
        dilation=int(dilation), closing=int(closing),
        crop=not full_image, crop_pad=int(crop_pad),
    )
    # Reconfigure the detector's mask source on the fly.
    detector.mask_source = mask_source

    out_dir = Path(tempfile.mkdtemp(prefix="mbr_ui_"))
    results = []
    paths = [Path(f) for f in files]
    for src in progress.tqdm(paths, desc="Cleaning"):
        try:
            img = io_utils.imread_unicode(src)
        except Exception as exc:  # noqa: BLE001
            gr.Warning(f"skip {src.name}: {exc}")
            continue
        result, _ = process_one(detector, inpainter, img, cfg)
        dst = out_dir / f"{src.stem}_clean.png"
        io_utils.imwrite_unicode(dst, result)
        results.append(str(dst))

    if not results:
        raise gr.Error("Nothing was produced (all inputs failed to decode).")

    zip_path = out_dir / "cleaned_pages.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            zf.write(r, Path(r).name)

    return results, str(zip_path)


def build_ui():
    title = "🫧 Manga Bubble & Text Remover"
    badge = f"GPU: {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else "⚠️ running on CPU (slow)"
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n{badge}")
        with gr.Row():
            with gr.Column(scale=1):
                files = gr.File(label="Pages", file_count="multiple", file_types=["image"], type="filepath")
                mask_source = gr.Dropdown(["mask", "lines", "both"], value="mask", label="Mask source ('both' = most aggressive)")
                threshold = gr.Slider(1, 200, value=30, step=1, label="Mask threshold (lower = catch fainter text)")
                dilation = gr.Slider(0, 25, value=5, step=1, label="Dilation px (grow mask over glyph edges)")
                closing = gr.Slider(0, 25, value=7, step=1, label="Closing px (solidify text blobs)")
                crop_pad = gr.Slider(16, 256, value=96, step=8, label="Crop context padding px")
                full_image = gr.Checkbox(value=False, label="Whole-page inpaint (slower, max context)")
                run = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Cleaned pages", columns=2, height=620)
                zip_out = gr.File(label="Download all (ZIP)")
        run.click(
            process,
            inputs=[files, mask_source, threshold, dilation, closing, crop_pad, full_image],
            outputs=[gallery, zip_out],
        )
    return demo


if __name__ == "__main__":
    if DEVICE == "cpu":
        print("[warn] CUDA not available — UI will run on CPU (very slow).")
    build_ui().launch()
