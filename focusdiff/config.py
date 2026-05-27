from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:
    torch = None


@dataclass
class FocusDiffConfig:
    seed: int = 42
    device: str = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    height: int = 512
    width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 10.0
    mask_scale: float = 0.1
    blur_kernel: int = 47
    blur_sigma: float = 8.0
    start_step: int = 7
    start_layer: int = 16
    torch_dtype: str = "float32"
    cache_dir: str = None
    local_files_only: bool = False


MODEL_PRESETS = {
    "sd15": {
        "model_path": "botp/stable-diffusion-v1-5",
        "model_type": "SD",
        "height": 512,
        "width": 512,
        "prediction_type": None,
    },
    "sd21": {
        "model_path": "sd2-community/stable-diffusion-2-1",
        "model_type": "SD",
        "height": 768,
        "width": 768,
        "prediction_type": "v_prediction",
    },
    "sdxl": {
        "model_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_type": "SDXL",
        "height": 1024,
        "width": 1024,
        "prediction_type": None,
    },
}
