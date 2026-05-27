# FocusDiff

FocusDiff is a tuning-free localized image editing pipeline based on target-aware refocusing attention. It inverts the original image and a background-blurred image, then transfers object structure from the focused branch while preserving the non-edited context.

## Versions

- `sd15`: original Stable Diffusion 1.5 FocusDiff backend.
- `sd21`: Stable Diffusion 2.1 backend with v-prediction DDIM inversion.
- `sdxl`: Stable Diffusion XL backend.

All versions use the same public entrypoint:

```bash
python run_focusdiff.py --version {sd15,sd21,sdxl}
```

The implementation keeps small backend adapters under `focusdiff/backends/` because SDXL and SD2.1 use different prompt encoding and inversion details, but the user-facing API is shared.

## Single Image

```bash
python run_focusdiff.py \
  --version sd21 \
  --image BenchMark_Blur/Images/23.jpg \
  --mask BenchMark_Blur/Masks/23.jpg \
  --prompt "a decorative lantern" \
  --output results/focusdiff_sd21.png
```

For SDXL:

```bash
python run_focusdiff.py \
  --version sdxl \
  --image path/to/image.png \
  --mask path/to/mask.png \
  --prompt "target prompt" \
  --output results/focusdiff_sdxl.png \
  --dtype float16
```

## Dataset

```bash
python run_focusdiff.py \
  --version sd15 \
  --dataset BenchMark_Blur \
  --output results
```

The dataset runner expects `annotates.json` plus `Images/` and `Masks/`. Each annotation should contain `img_name` and either `prompt` or `target_text`.
