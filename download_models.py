#!/usr/bin/env python3
"""Download the two model weights this project needs into ./models.

  * comictextdetector.pt.onnx  - text/bubble segmentation (comic-text-detector)
  * big-lama.pt                - TorchScript big-LaMa inpainter (IOPaint release)

Run once after installing dependencies:

    python download_models.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

MODELS = {
    "comictextdetector.pt.onnx": {
        "url": "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx",
        "md5": None,  # upstream does not publish one; size-checked instead
        "min_bytes": 40_000_000,
    },
    "big-lama.pt": {
        "url": "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
        "md5": "e3aa4aaa15225a33ec84f9f4bc47e500",
        "min_bytes": 180_000_000,
    },
}


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def already_ok(path: Path, spec: dict) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size < spec["min_bytes"]:
        return False
    if spec["md5"]:
        return md5sum(path) == spec["md5"]
    return True


def download(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    sys.stdout.write(f"\r  {dst.name}: {pct:3d}%  ({done >> 20}/{total >> 20} MiB)")
                    sys.stdout.flush()
    print()
    tmp.replace(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Download model weights for manga-bubble-remover")
    ap.add_argument("--models-dir", default="models", help="destination folder (default: ./models)")
    ap.add_argument("--force", action="store_true", help="re-download even if files look valid")
    args = ap.parse_args()

    out = Path(args.models_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, spec in MODELS.items():
        dst = out / name
        if not args.force and already_ok(dst, spec):
            print(f"[ok] {name} already present.")
            continue
        print(f"[get] {name}\n      {spec['url']}")
        try:
            download(spec["url"], dst)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] failed to download {name}: {exc}")
            return 1
        if spec["md5"] and md5sum(dst) != spec["md5"]:
            print(f"[error] checksum mismatch for {name}; delete it and retry.")
            return 1
        print(f"[ok] saved {dst} ({dst.stat().st_size >> 20} MiB)")

    print("\nAll models ready in", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
