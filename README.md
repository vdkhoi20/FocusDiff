# FocusDiff

**Target-Aware Refocusing for Tuning-Free Diffusion Editing**

[![Project Page](https://img.shields.io/badge/Project-Page-0ea5e9)](https://vdkhoi20.github.io/FocusDiff/)
[![Code](https://img.shields.io/badge/Code-GitHub-111827)](https://github.com/vdkhoi20/FocusDiff)
[![Benchmark](https://img.shields.io/badge/Benchmark-LIMB-16a34a)](#limb-benchmark)

FocusDiff is a tuning-free localized image editing framework for precise region-specific manipulation. It refocuses diffusion attention on the user-selected target by selectively blurring non-editing areas, then transfers object identity, structure, and appearance back to the edited image while preserving the surrounding context.

<p align="center">
  <img src="web/assets/teaser.webp" alt="FocusDiff teaser" width="900">
</p>

## Highlights

- Tuning-free localized text-guided editing with frozen diffusion models.
- Refocusing Cross-Attention for small or cluttered target objects.
- Context-Preserving Integration for stable backgrounds and reduced spillover.
- Shared API for Stable Diffusion 1.5, Stable Diffusion 2.1, and SDXL.
- 360-degree indoor panorama editing workflow for VR environments.
- LIMB benchmark with 30 multi-object images and 100 localized annotations.

## Project Page

The academic project page is available at:

https://vdkhoi20.github.io/FocusDiff/

The page source is the static `index.html` in this repository and uses the paper assets in `Images_in_Paper/`.

## Installation

```bash
git clone https://github.com/vdkhoi20/FocusDiff.git
cd FocusDiff
pip install -r requirements.txt
```

## Usage

Run a single image edit:

```bash
python run_focusdiff.py \
  --version sd21 \
  --image LIMB/Images/23.jpg \
  --mask LIMB/Masks/23.jpg \
  --prompt "a decorative lantern" \
  --output results/focusdiff_sd21.png
```

Available backbones:

```bash
python run_focusdiff.py --version sd15
python run_focusdiff.py --version sd21
python run_focusdiff.py --version sdxl
```

Run on the LIMB dataset:

```bash
python run_focusdiff.py \
  --version sd15 \
  --dataset LIMB \
  --output results
```

The dataset runner expects `annotates.json`, `Images/`, and `Masks/`. Each annotation should contain `img_name` and either `prompt` or `target_text`.

## Quantitative Results

| Method | CLIPScore ↑ | LPIPS ↓ |
| --- | ---: | ---: |
| MasaCtrl | 20.12 | 0.280 |
| Blended-Diffusion | 27.43 | 0.156 |
| DiffEdit | 27.75 | 0.148 |
| LEDITS++ | 32.76 | 0.103 |
| CPAM | 33.45 | 0.101 |
| **FocusDiff-SD1.5** | **35.85** | **0.099** |
| **FocusDiff-SD2.1** | **35.61** | **0.068** |
| **FocusDiff-SDXL** | **36.48** | **0.064** |

## LIMB Benchmark

LIMB is a localized image manipulation benchmark curated from PIE-Bench for fine-grained region-specific editing. It contains 30 multi-object images and 100 localized editing annotations, including challenging small-object cases.

## BibTeX

```bibtex
@inproceedings{vo2026focusdiff,
  title = {Toward 360-Degree Indoor Panorama Editing via Tuning-Free Diffusion Model with Refocusing Cross-Attention},
  author = {Vo, Dinh-Khoi and Le-Hinh, Nhut-Thanh and Huynh, Viet-Tham and Nguyen, Tam V. and Tran, Minh-Triet and Le, Trung-Nghia},
  booktitle = {International Conference on Computational Collective Intelligence},
  year = {2026}
}
```
