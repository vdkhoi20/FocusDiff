import argparse
import importlib.util

from focusdiff import FocusDiffConfig


REQUIRED_MODULES = ["torch", "diffusers", "transformers", "PIL", "cv2", "einops"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run FocusDiff on one image or an annotated dataset.")
    parser.add_argument("--version", choices=["sd15", "sd21", "sdxl"], default="sd15")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image", default=None, help="Single source image path.")
    parser.add_argument("--mask", default=None, help="Single binary mask path.")
    parser.add_argument("--prompt", default=None, help="Target edit prompt for single-image mode.")
    parser.add_argument("--output", default="results/focusdiff.png")
    parser.add_argument("--dataset", default=None, help="Dataset root containing annotates.json, Images, Masks.")
    parser.add_argument("--annot-file", default="annotates.json")
    parser.add_argument("--image-dir", default="Images")
    parser.add_argument("--mask-dir", default="Masks")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=10.0)
    parser.add_argument("--mask-scale", type=float, default=0.1)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check-env", action="store_true", help="Check Python dependencies and exit.")
    return parser.parse_args()


def check_env():
    missing = []
    for module in REQUIRED_MODULES:
        ok = importlib.util.find_spec(module) is not None
        print(f"{module}: {'ok' if ok else 'missing'}")
        if not ok:
            missing.append(module)
    if missing:
        raise SystemExit(f"Missing dependencies: {', '.join(missing)}")


def main():
    args = parse_args()
    if args.check_env:
        check_env()
        return

    if not args.dataset and not (args.image and args.mask and args.prompt):
        raise SystemExit("Single-image mode needs --image, --mask, and --prompt, or use --dataset.")

    from focusdiff import FocusDiff

    cfg = FocusDiffConfig(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        mask_scale=args.mask_scale,
        torch_dtype=args.dtype,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    focusdiff = FocusDiff(args.version, config=cfg, model_path=args.model_path, device=args.device)

    if args.dataset:
        focusdiff.run_dataset(
            args.dataset,
            annot_file=args.annot_file,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output,
            limit=args.limit,
        )
        return

    focusdiff.edit_image(args.image, args.mask, args.prompt, args.output)


if __name__ == "__main__":
    main()
