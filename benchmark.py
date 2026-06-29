#!/usr/bin/env python3
"""Benchmark the pipeline: per-stage latency, throughput, and peak VRAM.

Examples
--------
    # benchmark on real sample pages
    python benchmark.py -i ./samples

    # no images handy? generate synthetic 2048x1440 pages
    python benchmark.py --synthetic 30

    # compare crop vs whole-image inpainting
    python benchmark.py -i ./samples --full-image
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mbr import imgproc, io_utils
from mbr.detector import TextDetector
from mbr.inpainter import LamaInpainter
from mbr.pipeline import Config


def _synthetic_pages(n: int, h: int, w: int):
    """Make ``n`` fake manga-ish pages with a couple of black 'text' blocks."""
    pages = []
    for i in range(n):
        img = np.full((h, w, 3), 235, np.uint8)
        # screentone-ish gradient so inpainting has something to reconstruct
        grad = np.linspace(180, 255, w, dtype=np.uint8)
        img[:] = np.dstack([np.tile(grad, (h, 1))] * 3)
        # two solid "text" rectangles
        img[h // 6 : h // 6 + 120, w // 8 : w // 8 + 380] = 20
        img[int(h * 0.6) : int(h * 0.6) + 90, int(w * 0.5) : int(w * 0.5) + 300] = 20
        pages.append(img)
    return pages


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark manga-bubble-remover")
    ap.add_argument("-i", "--input", nargs="+", help="sample image(s)/folder(s)")
    ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic pages instead")
    ap.add_argument("--syn-size", default="2048x1440", help="synthetic page WxH")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--warmup", type=int, default=3, help="untimed warmup iterations")
    ap.add_argument("--dilation", type=int, default=5)
    ap.add_argument("--mask-threshold", type=int, default=30)
    ap.add_argument("--mask-source", default="mask", choices=["mask", "lines", "both"])
    ap.add_argument("--full-image", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[error] CUDA not available; this benchmark targets GPU.", file=sys.stderr)
        return 2

    # Gather images.
    if args.synthetic > 0:
        w, h = (int(x) for x in args.syn_size.lower().split("x"))
        images = _synthetic_pages(args.synthetic, h, w)
        label = f"{args.synthetic} synthetic {w}x{h} pages"
    elif args.input:
        paths = io_utils.collect_images(args.input)
        if not paths:
            print("[error] no images found.", file=sys.stderr)
            return 2
        images = [io_utils.imread_unicode(p) for p in paths]
        label = f"{len(images)} real pages"
    else:
        print("[error] pass -i <images> or --synthetic N", file=sys.stderr)
        return 2

    models = Path(args.models_dir)
    det = models / "comictextdetector.pt.onnx"
    lama = models / "big-lama.pt"
    if not det.exists() or not lama.exists():
        print(f"[error] missing models in {models}; run download_models.py", file=sys.stderr)
        return 2

    cfg = Config(
        detector_model=str(det), lama_model=str(lama), device=args.device,
        mask_source=args.mask_source, mask_threshold=args.mask_threshold,
        dilation=args.dilation, crop=not args.full_image,
    )

    if args.device == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print(f"[load] models...")
    detector = TextDetector(cfg.detector_model, device=cfg.device, mask_source=cfg.mask_source)
    inpainter = LamaInpainter(cfg.lama_model, device=cfg.device)

    # Warmup (autotune + first-call graph build) on the first image.
    print(f"[warmup] {args.warmup} iter(s)...")
    for _ in range(args.warmup):
        prob = detector.detect(images[0])
        mask = imgproc.refine_mask(prob, threshold=cfg.mask_threshold, dilation=cfg.dilation, closing=cfg.closing)
        if mask.max() > 0:
            inpainter.inpaint(images[0], mask, crop=cfg.crop, crop_pad=cfg.crop_pad)
    _sync(cfg.device)

    det_ms, inp_ms, tot_ms = [], [], []
    print(f"[run] timing {label} (crop={cfg.crop})...")
    for img in images:
        _sync(cfg.device)
        t0 = time.perf_counter()
        prob = detector.detect(img)
        mask = imgproc.refine_mask(prob, threshold=cfg.mask_threshold, dilation=cfg.dilation, closing=cfg.closing)
        _sync(cfg.device)
        t1 = time.perf_counter()
        if mask.max() > 0:
            inpainter.inpaint(img, mask, crop=cfg.crop, crop_pad=cfg.crop_pad)
        _sync(cfg.device)
        t2 = time.perf_counter()
        det_ms.append((t1 - t0) * 1000)
        inp_ms.append((t2 - t1) * 1000)
        tot_ms.append((t2 - t0) * 1000)

    def summarize(name, xs):
        xs_s = sorted(xs)
        p50 = statistics.median(xs_s)
        p95 = xs_s[min(len(xs_s) - 1, int(0.95 * len(xs_s)))]
        print(f"  {name:10s} mean {statistics.mean(xs):7.1f} ms   p50 {p50:7.1f}   p95 {p95:7.1f}")

    total_s = sum(tot_ms) / 1000
    print("\n=== results ===")
    summarize("detect", det_ms)
    summarize("inpaint", inp_ms)
    summarize("total", tot_ms)
    print(f"  throughput {len(images) / total_s:6.2f} img/s   "
          f"({total_s:.1f}s for {len(images)} imgs)")
    if cfg.device == "cuda":
        peak = torch.cuda.max_memory_allocated() / (1 << 20)
        print(f"  peak VRAM  {peak:7.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
