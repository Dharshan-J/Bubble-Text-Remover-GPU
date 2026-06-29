# 🫧 Manga Bubble & Text Remover (GPU)

High-performance, **GPU-only** batch tool that erases speech-bubble text *and*
free-floating text from manga / comic pages and cleanly reconstructs the artwork
behind it. Point it at a folder of 300+ images and it writes the cleaned pages
to a new folder, preserving your sub-folder structure.

It uses the same models the best open-source manga translators rely on, wired
into an optimized batch pipeline:

| Stage | Model | Runs on |
|-------|-------|---------|
| **Text/bubble detection** | [`comic-text-detector`](https://github.com/dmMaze/comic-text-detector) (ONNX) | GPU via `onnxruntime-gpu` (CUDA) |
| **Inpainting / fill** | [`big-LaMa`](https://github.com/advimman/lama) (TorchScript) | GPU via PyTorch (CUDA) |

**Why this is better than the YOLOv8 + ADetailer reference repo:** comic-text-detector
produces a *pixel-level* text mask (not just bounding boxes), so it removes text
sitting **on top of artwork** without erasing a big rectangle of the drawing.
big-LaMa then rebuilds screentone / gradients far more convincingly than classic
inpainting. The pipeline also overlaps disk I/O with GPU compute and inpaints
only the region around the text, so it's fast on large batches.

---

## Requirements

### Hardware
- **NVIDIA GPU with CUDA** (developed/tuned for an **RTX 4090**, 24 GB).
  Any RTX 20-series or newer works; ~6 GB VRAM is enough in the default
  crop-inpaint mode.

### Software
- **NVIDIA driver** new enough for CUDA 12.x (any recent Game Ready / Studio driver).
  You do **not** need a system-wide CUDA Toolkit — the PyTorch and ONNX Runtime
  wheels bundle their own CUDA runtime.
- **Python 3.9 – 3.11** (64-bit).
- ~250 MB of disk for the two model files.

### Python packages
- `torch` (CUDA build — installed separately, see below)
- `onnxruntime-gpu`, `opencv-python-headless`, `numpy<2`, `tqdm`, `requests`
  (in `requirements.txt`)

---

## Setup

### 1. Get the code & create a virtual environment

```bash
cd manga-bubble-remover

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install dependencies

**Easiest (recommended) — one pinned, GPU-ready command** installs everything
*including* the CUDA build of PyTorch and the web-UI deps:

```bash
pip install --upgrade pip
pip install -r requirements-cu121.txt
```

If you use this, **skip step 3** — torch is already installed. Then go to step 4.

**On a CUDA 13 machine?** Use the CUDA-13 lockfile instead:
```bash
pip install --upgrade pip
pip install -r requirements-cu130.txt
```
This installs **native CUDA-13 PyTorch** plus the **CUDA-12 `onnxruntime-gpu`**
(which runs fine on a CUDA-13 driver, since NVIDIA drivers are backward
compatible — native CUDA-13 ONNX Runtime is nightly-only today). Skip step 3.

> Tip: a CUDA-13 driver also runs the CUDA-12 lockfile (`requirements-cu121.txt`)
> without issue, if you'd rather keep the whole stack on CUDA 12.

<details>
<summary>Or install manually (base deps only)</summary>

```bash
pip install --upgrade pip
pip install -r requirements.txt   # then do step 3 to add torch
```
</details>

### 3. Install the CUDA build of PyTorch  ⚠️ important (skip if you used the lockfile)

Install PyTorch **separately** with the CUDA index so you don't get the CPU-only
wheel. For an RTX 4090, CUDA 12.1 wheels are the sweet spot:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> Prefer the newest stable CUDA wheels (e.g. `cu124`) if you like — the 4090
> supports them all. Just keep `onnxruntime-gpu` and `torch` on compatible
> CUDA major versions (both on CUDA 12.x is fine).

### 4. Verify the GPU is visible

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import onnxruntime as ort; print('ORT providers:', ort.get_available_providers())"
```

You should see `CUDA: True ... NVIDIA GeForce RTX 4090` and
`CUDAExecutionProvider` in the ONNX Runtime providers list.

### 5. Download the models

```bash
python download_models.py
```

This fetches `comictextdetector.pt.onnx` and `big-lama.pt` into `./models/`
(verified by size / MD5). Re-run any time; it skips files already present.

---

## Usage

Clean a whole folder (recurses into sub-folders by default). Output goes to a
sibling `<folder>_cleaned` directory unless you pass `-o`:

```bash
python run.py -i ./chapter1
# -> ./chapter1_cleaned/...
```

Pick the output folder explicitly:

```bash
python run.py -i ./manga/chapter1 -o ./done/chapter1
```

Process specific files (any mix of files and folders works):

```bash
python run.py -i page01.jpg page02.png cover.webp -o ./out
```

A 300-image batch is just a folder:

```bash
python run.py -i ./big_batch -o ./big_batch_clean
```

Supported formats: `.jpg .jpeg .png .webp .bmp .tif .tiff`.

### Useful options

| Flag | Default | What it does |
|------|---------|--------------|
| `-i, --input` | — | Files and/or folders to process (required) |
| `-o, --output` | `<input>_cleaned` | Output folder |
| `--dilation N` | `5` | Grow the mask N px so glyph edges/halos are fully erased. Bump to `9–13` if faint outlines remain |
| `--mask-threshold N` | `30` | 0–255; **lower** catches fainter text |
| `--mask-source` | `mask` | `mask`, `lines`, or `both` (most aggressive) |
| `--closing N` | `7` | Fuse nearby strokes into solid blobs before filling |
| `--full-image` | off | Inpaint the whole page (slower, max context) instead of cropping around text |
| `--crop-pad N` | `96` | Context padding (px) kept around text when cropping |
| `--overwrite` | off | Re-process files already present in the output |
| `--no-recursive` | off | Don't descend into sub-folders |
| `--io-workers N` | `4` | Decode/encode threads overlapped with GPU work |
| `--jpeg-quality N` | `95` | Quality for `.jpg` / `.webp` output |
| `-v, --verbose` | off | Print provider info, model output shapes, full tracebacks |

### Tuning tips
- **Text left behind?** Lower `--mask-threshold` (e.g. `20`) and/or set
  `--mask-source both`.
- **Faint glyph outlines remain?** Increase `--dilation` (e.g. `11`).
- **Artwork over-erased / smudged?** Reduce `--dilation` and keep crop mode on.
- **Best-quality fills regardless of speed?** Add `--full-image`.

---

## Web UI (optional)

Prefer drag-and-drop over the terminal? Launch the Gradio app:

```bash
pip install gradio            # already included if you used requirements-cu121.txt
python app.py                 # open the printed http://127.0.0.1:7860
```

Drop in one or many pages, adjust the sliders (mask source/threshold/dilation,
crop padding, whole-page toggle), hit **Run** — cleaned pages appear in the
gallery and you can download them all as a ZIP. Models load once on first run.

---

## Benchmark (optional)

Measure per-stage latency, throughput, and peak VRAM on your 4090:

```bash
# on your own sample pages
python benchmark.py -i ./samples

# or with generated synthetic pages (no images needed)
python benchmark.py --synthetic 30 --syn-size 2048x1440

# compare crop vs whole-image inpainting
python benchmark.py -i ./samples --full-image
```

Output reports mean / p50 / p95 for **detect** and **inpaint** stages, overall
**img/s**, and **peak VRAM**. Warmup iterations (cuDNN/ORT autotune) are excluded
from timing.

---

## How it works

```
 page.jpg
    │
    ▼  decode (thread pool, prefetched)
 ┌──────────────────────────────┐
 │ comic-text-detector (ONNX/GPU)│  -> soft text mask
 └──────────────────────────────┘
    │  threshold + close + dilate
    ▼
 ┌──────────────────────────────┐
 │ big-LaMa (TorchScript/GPU)    │  -> fills only the masked pixels
 │ (crops to text region + pad)  │     (untouched art is byte-identical)
 └──────────────────────────────┘
    │
    ▼  encode (thread pool)
 chapter1_cleaned/page.jpg
```

Performance choices:
- **GPU end-to-end** — detection on `onnxruntime-gpu` (CUDA), inpainting on CUDA PyTorch.
- **TF32 + cuDNN autotuning** enabled for the 4090.
- **Crop-inpaint**: only the bounding region around detected text (plus context)
  is sent to LaMa, then composited back — a large speed/VRAM win on sparse pages.
  Pages whose text covers a big fraction of the page fall back to whole-image fill.
- **Overlapped I/O**: pages are decoded ahead and encoded behind the GPU stages.
- Pages with **no detected text are copied through** unchanged.

> **Precision note:** big-LaMa's scripted graph is run in fp32 (its
> Fourier-convolution blocks are unstable under fp16). On a 4090 this is
> effectively as fast thanks to TF32, while staying numerically safe.

---

## Project layout

```
manga-bubble-remover/
├── run.py               # CLI entrypoint
├── app.py               # Gradio web UI
├── benchmark.py         # latency / throughput / VRAM benchmark
├── download_models.py   # fetches the two model weights into ./models
├── requirements.txt         # base deps (torch installed separately)
├── requirements-cu121.txt   # pinned one-shot install incl. CUDA torch + gradio
├── README.md
├── models/              # comictextdetector.pt.onnx, big-lama.pt (downloaded)
└── mbr/
    ├── detector.py      # comic-text-detector ONNX wrapper (CUDA)
    ├── inpainter.py     # big-LaMa TorchScript wrapper (CUDA)
    ├── pipeline.py      # batch orchestration + overlapped I/O
    ├── imgproc.py       # letterbox, mask refine, crop, modulo-pad
    └── io_utils.py      # image discovery + unicode-safe read/write
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA is not available to PyTorch` | You installed the CPU wheel. Reinstall: `pip uninstall torch` then the `cu121` command in step 3. |
| `onnxruntime did not initialise the CUDA execution provider` | You have `onnxruntime` instead of `onnxruntime-gpu`. `pip uninstall onnxruntime onnxruntime-gpu` then `pip install onnxruntime-gpu`. |
| `CUDA out of memory` | Keep crop mode (don't use `--full-image`), lower `--crop-pad`, or close other GPU apps. The 4090 handles full pages fine; this matters more on smaller cards. |
| Non-ASCII / Japanese filenames fail | Already handled (unicode-safe read/write), but make sure the output drive allows those characters. |
| Slow first image | Normal — cuDNN/ORT autotune on the first run, then it speeds up. |
| `missing model file(s)` | Run `python download_models.py`. |

---

## Credits / models
- Text detection: **comic-text-detector** — https://github.com/dmMaze/comic-text-detector
  (weights via [manga-image-translator releases](https://github.com/zyddnys/manga-image-translator/releases))
- Inpainting: **LaMa** — https://github.com/advimman/lama
  (big-LaMa TorchScript via [IOPaint / Sanster models](https://github.com/Sanster/models/releases))

Use responsibly and respect the copyright of the source material.
