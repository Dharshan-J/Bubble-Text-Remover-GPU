#!/usr/bin/env python3
"""manga-bubble-remover CLI.

Examples
--------
    # clean an entire folder (and sub-folders) -> ./chapter1_cleaned
    python run.py -i ./chapter1

    # several explicit files into a chosen folder
    python run.py -i a.jpg b.png c.webp -o ./out

    # 300+ pages, aggressive mask growth, keep originals' structure
    python run.py -i ./manga --dilation 9 --recursive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from mbr.pipeline import Config, run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GPU batch removal of manga/comic text and speech bubbles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", nargs="+", required=True,
                   help="image file(s) and/or folder(s) to process")
    p.add_argument("-o", "--output", default=None,
                   help="output folder (default: <input>_cleaned)")
    p.add_argument("--models-dir", default="models", help="folder containing the model weights")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                   help="cpu is for debugging only; this tool targets GPU")

    p.add_argument("--no-recursive", action="store_true", help="do not descend into sub-folders")
    p.add_argument("--overwrite", action="store_true", help="re-process files that already exist in output")

    # mask controls
    p.add_argument("--mask-source", default="mask", choices=["mask", "lines", "both"],
                   help="which detector head to use ('both' is the most aggressive)")
    p.add_argument("--mask-threshold", type=int, default=30,
                   help="0-255; lower = catch fainter text")
    p.add_argument("--dilation", type=int, default=5,
                   help="grow mask outward (px) to fully cover glyph edges")
    p.add_argument("--closing", type=int, default=7,
                   help="morphological close (px) to solidify text blobs")

    # inpaint controls
    p.add_argument("--full-image", action="store_true",
                   help="inpaint the whole page instead of cropping around text")
    p.add_argument("--crop-pad", type=int, default=96,
                   help="context padding (px) around text when cropping")

    # io / perf
    p.add_argument("--io-workers", type=int, default=4, help="threads for image decode/encode")
    p.add_argument("--jpeg-quality", type=int, default=95, help="quality for .jpg/.webp output")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print(
            "[error] CUDA is not available to PyTorch. This tool is GPU-only.\n"
            "        Install a CUDA build of torch (see README) or pass --device cpu "
            "for a (slow) debug run.",
            file=sys.stderr,
        )
        return 2
    if args.device == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory >> 30} GiB)")

    models = Path(args.models_dir)
    det = models / "comictextdetector.pt.onnx"
    lama = models / "big-lama.pt"
    missing = [str(m) for m in (det, lama) if not m.exists()]
    if missing:
        print(f"[error] missing model file(s): {missing}\n"
              f"        run: python download_models.py --models-dir {args.models_dir}",
              file=sys.stderr)
        return 2

    cfg = Config(
        detector_model=str(det),
        lama_model=str(lama),
        device=args.device,
        output_dir=Path(args.output) if args.output else None,
        recursive=not args.no_recursive,
        overwrite=args.overwrite,
        mask_source=args.mask_source,
        mask_threshold=args.mask_threshold,
        dilation=args.dilation,
        closing=args.closing,
        crop=not args.full_image,
        crop_pad=args.crop_pad,
        io_workers=args.io_workers,
        jpeg_quality=args.jpeg_quality,
        verbose=args.verbose,
    )
    stats = run(args.input, cfg)
    return 1 if stats.errors and (stats.processed + stats.no_text) == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
