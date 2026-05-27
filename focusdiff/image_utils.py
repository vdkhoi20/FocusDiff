from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ImagePath = Union[str, Path]


def load_rgb(path: ImagePath, size: Tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize(size, Image.LANCZOS)


def pil_to_model_tensor(image: Image.Image, device: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    array = np.asarray(image).astype(np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor / 127.5 - 1.0).to(device=device, dtype=dtype)


def load_binary_mask(path: ImagePath, device: str, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    if size is not None:
        mask = mask.resize(size, Image.NEAREST)
    array = np.asarray(mask)
    if array.max() == 0:
        raise ValueError(f"Mask is empty: {path}")
    return torch.from_numpy((array > 127).astype(np.float32)).to(device)


def expand_mask(mask: torch.Tensor, scale: float) -> torch.Tensor:
    object_size = torch.sum(mask)
    kernel_size = int(torch.sqrt(object_size).item() * scale)
    if kernel_size <= 1:
        return mask.float()
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    expanded = F.conv2d(mask.float()[None, None], kernel, padding=kernel_size // 2)
    return (expanded.squeeze() > 0).float()


def blur_background(image: Image.Image, object_mask: torch.Tensor, kernel: int, sigma: float) -> Image.Image:
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    image_np = np.asarray(image).astype(np.float32) / 255.0
    mask_np = object_mask.detach().float().cpu().numpy()[..., None]
    blurred = cv2.GaussianBlur(image_np, (kernel, kernel), sigma)
    result = image_np * mask_np + blurred * (1.0 - mask_np)
    return Image.fromarray((result.clip(0, 1) * 255).astype(np.uint8))


def safe_name(text: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip()]
    return "".join(keep).strip("_")[:160] or "result"
