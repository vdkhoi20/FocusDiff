import os
import torch
import numpy as np
import cv2 
from pathlib import Path

import torch.nn.functional as F

from torchvision.utils import save_image, make_grid
from torchvision.io import read_image

from diffusers import DDIMScheduler
from focusdiff.backends.sd15.diffuser_utils import OIICtrlPipeline
from focusdiff.backends.sd15.OIIctrl import OIISelfAttentionControlMask
from focusdiff.backends.sd15.config import Config as cfg
from focusdiff.backends.sd15.OIIctrl_utils import regiter_attention_editor_diffusers, AttentionBase
from focusdiff.seed import seed_everything
from torchvision.transforms.functional import to_pil_image


class ImageInversionProcessor:
    """
    Class to handle the image inversion and generation pipeline.

    It loads the model, preprocesses input data, and generates images based on the provided parameters.
    """

    def __init__(self, config=None, model_path="sd-legacy/stable-diffusion-v1-5"):
        """
        Initializes the ImageInversionProcessor with configuration and model path.

        Args:
            config (object): The configuration object containing parameters like DEVICE, MAX_STEP, etc.
            model_path (str): The path to the Stable Diffusion model.
        """
        seed_everything(42)

        self.cfg = config if config else cfg
        self.device = self.cfg.DEVICE
        print(f"Using device: {self.device}")
        self.model_path = model_path

        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False
        )
        self.model = OIICtrlPipeline.from_pretrained(
            self.model_path, scheduler=self.scheduler
        ).to(self.device)

    def _extract_object_mask(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extracts the object mask from the input mask image.
        
        Args:
            image (torch.Tensor): Image tensor with shape [C, H, W].

        Returns:
            torch.Tensor: Binary mask with shape [H, W].
        """
        if image.shape[0] > 2:
            mask_channel = image[2]
        elif image.shape[0] == 1:
            mask_channel = image.squeeze(0)
        else:
            raise ValueError("Unsupported image shape for mask extraction.")

        mask_channel = mask_channel.float()
        object_mask = (mask_channel > 0.0).float()
        return object_mask

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalizes an image tensor to the model's required format.
        Args:
            image (torch.Tensor): Image tensor with shape [C, H, W].
        Returns:
            torch.Tensor: Normalized image tensor.
        """
        if image.shape[0] > 3:
            image = image[:3]
        image = image.unsqueeze_(0).float() / 127.5 - 1.0
        image = image.to(self.device)
        return image

    def _apply_gaussian_blur_numpy(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Applies Gaussian blur to the masked region of the image using OpenCV.
        Args:
            image (torch.Tensor): Input image tensor (C, H, W).
            mask (torch.Tensor): Binary mask tensor (H, W).
        Returns:
            torch.Tensor: The blurred image tensor.
        """
        image_np = image.clone().permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0
        
        # Corrected: Squeeze mask to a 2D tensor if it has extra dimensions
        if mask.dim() > 2:
            mask = mask.squeeze()
        mask_np = mask.clone().cpu().numpy().astype(np.float32)

        blurred_image_np = cv2.GaussianBlur(image_np, (47, 47), 8)
        
        mask_np = np.expand_dims(mask_np, axis=2) 
        
        result_np = image_np * mask_np + blurred_image_np * (1 - mask_np)
        
        result_tensor = torch.tensor(result_np, dtype=torch.float32).permute(2, 0, 1).to(self.device)
        return result_tensor * 255.0

    def _expand_mask(self, mask, scale=0.15):
        """
        Mở rộng mask bằng cách sử dụng phép tích chập (convolution) để làm giãn mask.
        Args:
            mask (torch.Tensor): Mask nhị phân đầu vào.
            scale (float): Hệ số tỷ lệ để xác định kích thước kernel.
        Returns:
            torch.Tensor: Mask đã được mở rộng.
        """
        object_size = torch.sum(mask)
        kernel_size = int(torch.sqrt(object_size).item() * scale)
        if (kernel_size == 0): return mask 
        
        # Sửa: Loại bỏ các chiều thừa trước khi thêm chiều batch và channel.
        # Đảm bảo mask là 2D (H, W) trước khi unsqueeze
        source_mask_tensor = mask.clone().detach().squeeze().unsqueeze(0).unsqueeze(0).float()
        
        dilation = torch.ones(1, 1, kernel_size, kernel_size).to(source_mask_tensor.device) 
        
        expanded_mask_tensor = F.conv2d(source_mask_tensor, dilation, padding=kernel_size // 2)
        expanded_mask_tensor = torch.where(expanded_mask_tensor > 0, torch.tensor(1.0).to(source_mask_tensor.device), torch.tensor(0.0).to(source_mask_tensor.device))
        expanded_mask = expanded_mask_tensor.squeeze().byte()

        return expanded_mask
    
    def _crop_image_and_mask(self, image: torch.Tensor, mask: torch.Tensor):
        """
        Crops the image and mask to a square region around the mask's bounding box.
        Args:
            image (torch.Tensor): Original image (C, H, W).
            mask (torch.Tensor): Binary mask (H, W).
        Returns:
            tuple: (cropped_image, cropped_mask, bbox_coords)
        """
        _, h, w = image.shape
        coords = torch.nonzero(mask)
        if coords.numel() == 0:
            center_x, center_y = w // 2, h // 2
            crop_size = min(512, h, w)
            start_x = max(0, center_x - crop_size // 2)
            end_x = start_x + crop_size
            start_y = max(0, center_y - crop_size // 2)
            end_y = start_y + crop_size

            if end_x > w: start_x, end_x = w - crop_size, w
            if end_y > h: start_y, end_y = h - crop_size, h
            start_x, start_y = max(0, start_x), max(0, start_y)
        else:
            y_min_tensor, x_min_tensor = coords.min(dim=0)[0]
            y_max_tensor, x_max_tensor = coords.max(dim=0)[0]
            y_min, x_min = y_min_tensor.item(), x_min_tensor.item()
            y_max, x_max = y_max_tensor.item(), x_max_tensor.item()

            mask_h, mask_w = y_max - y_min, x_max - x_min
            raw_crop_size = int(max(mask_h, mask_w) * 1.5)
            crop_size = max(256, min(raw_crop_size, 768))

            center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
            start_x, start_y = center_x - crop_size // 2, center_y - crop_size // 2

            start_x = max(0, min(start_x, w - crop_size))
            start_y = max(0, min(start_y, h - crop_size))
            end_x, end_y = start_x + crop_size, start_y + crop_size

        cropped_image = image[:, start_y:end_y, start_x:end_x]
        cropped_mask = mask[start_y:end_y, start_x:end_x]
        bbox_coords = (start_y, start_x, end_y, end_x)
        return cropped_image, cropped_mask, bbox_coords

    def _paste_edited_image_back(self, original_image: torch.Tensor, edited_cropped_image: torch.Tensor, bbox_coords: tuple):
        """
        Pastes the edited image back into the original image.
        Args:
            original_image (torch.Tensor): Original image (C, H, W).
            edited_cropped_image (torch.Tensor): Edited image (C, 512, 512).
            bbox_coords (tuple): Bounding box coordinates (start_y, start_x, end_y, end_x) of the original crop.
        Returns:
            torch.Tensor: The updated original image.
        """
        start_y, start_x, end_y, end_x = bbox_coords
        target_h, target_w = end_y - start_y, end_x - start_x
        
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Invalid paste region size: H={target_h}, W={target_w}")

        resized_edited_image = F.interpolate(
            edited_cropped_image.unsqueeze(0), 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        final_image = original_image.clone()
        final_image[:, start_y:end_y, start_x:end_x] = resized_edited_image
        return final_image

    def _preprocess_data_for_image_generation(self, source_image: torch.Tensor, source_mask: torch.Tensor, target_prompt: str, img_name: str):
        """
        Preprocessing step: loads, blurs, normalizes, and inverts the image.
        """
        image_object = self._apply_gaussian_blur_numpy(source_image, source_mask).to(self.device)
        source_mask_expanded = self._expand_mask(source_mask, self.cfg.SCALE_MASK).to(self.device)
        
        source_image_norm = self._normalize_image(source_image).to(self.device)
        image_object_norm = self._normalize_image(image_object).to(self.device)
        
        editor = AttentionBase()
        regiter_attention_editor_diffusers(self.model, editor)
        
        start_code_object, intermediate_objects = self.model.invert(
            image_object_norm,
            prompt="", guidance_scale=0, num_inference_steps=self.cfg.MAX_STEP,
            return_intermediates=True, DEVICE=self.device
        )

        start_code, intermediates = self.model.invert(
            source_image_norm,
            prompt="", guidance_scale=0, num_inference_steps=self.cfg.MAX_STEP,
            return_intermediates=True, DEVICE=self.device
        )

        latents = torch.cat([
            start_code.clone(), start_code.clone(), 
            start_code_object.clone(), start_code_object.clone()
        ])
        
        return {
            "source_mask_expanded": source_mask_expanded,
            "intermediate_objects": intermediate_objects,
            "intermediates": intermediates,
            "latents": latents,
            "target_prompt": target_prompt,
            "img_name": img_name
        }

    def _generate_image_from_preprocessed_data(self, preprocessed_data: dict, out_dir: Path, DoErase: bool= False):
        """
        Main image generation step: uses preprocessed data to generate and save images.
        """
        editor = OIISelfAttentionControlMask(
            start_step=self.cfg.STEP_QUERY, start_layer=self.cfg.LAYER_QUERY,
            mask=preprocessed_data["source_mask_expanded"], total_steps=self.cfg.MAX_STEP,
            DoErase=DoErase
        )
        regiter_attention_editor_diffusers(self.model, editor)

        prompts = ["", "", "", "", preprocessed_data["target_prompt"]]
        image_result = self.model(
            prompts, latents=preprocessed_data["latents"], ref_original=None, 
            ref_intermediate_objects=preprocessed_data["intermediate_objects"],
            ref_intermediates=preprocessed_data["intermediates"],
            guidance_scale=self.cfg.GUIDANCE_SCALE,
            num_inference_steps=self.cfg.MAX_STEP,
            return_intermediates=False,
            DEVICE=self.device
        )

        edited_image = image_result[0]
        
        # image_compose = [
        #     edited_image, image_result[1], image_result[2], image_result[3]
        # ]
        # order = preprocessed_data['img_name'].split(".")[0]
        # out_path = out_dir / f"{order}_debug_{preprocessed_data['target_prompt'].replace(' ', '_')}.png"
        # out_images = make_grid(image_compose, nrow=len(image_compose))
        # save_image(out_images, out_path)
        # print(f"Debug images are saved in {out_path}")

        return edited_image

    def run_pipeline(self, image_path: str, mask_path: str, target_prompt: str, output_dir: str = "results", DoErase: bool= False):
        out_dir = Path(output_dir) / "_panorama"
        os.makedirs(out_dir, exist_ok=True)
        
        # Corrected: Call read_image from torchvision.io directly.
        original_image = read_image(str(Path(image_path)))
        mask_image = read_image(str(Path(mask_path))) 
        
        cropped_image, cropped_mask, bbox_coords = self._crop_image_and_mask(original_image, self._extract_object_mask(mask_image))
        
        # cropped_image_path = out_dir / f"cropped_image_{Path(image_path).stem}.png"
        # save_image(cropped_image / 255.0, cropped_image_path)
        # print(f"Cropped image saved to {cropped_image_path}")

        # cropped_mask_path = out_dir / f"cropped_mask_{Path(image_path).stem}.png"
        # save_image(cropped_mask.unsqueeze(0), cropped_mask_path)
        # print(f"Cropped mask saved to {cropped_mask_path}")
        
        model_input_size = 512
        resized_cropped_image = F.interpolate(cropped_image.unsqueeze(0), size=(model_input_size, model_input_size), mode='bilinear', align_corners=False).squeeze(0)
        resized_cropped_mask = F.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=(model_input_size, model_input_size), mode='nearest').squeeze(0)
        
        # resized_cropped_image_path = out_dir / f"resized_cropped_image_{Path(image_path).stem}.png"
        # save_image(resized_cropped_image / 255.0, resized_cropped_image_path)
        # print(f"Resized cropped image (for model input) saved to {resized_cropped_image_path}")

        # resized_cropped_mask_path = out_dir / f"resized_cropped_mask_{Path(image_path).stem}.png"
        # save_image(resized_cropped_mask.unsqueeze(0), resized_cropped_mask_path)
        # print(f"Resized cropped mask (for model input) saved to {resized_cropped_mask_path}")

        preprocessed_data = self._preprocess_data_for_image_generation(
            resized_cropped_image,
            resized_cropped_mask,
            target_prompt,
            Path(image_path).name
        )
        
        edited_cropped_image = self._generate_image_from_preprocessed_data(preprocessed_data, out_dir, DoErase)
        
        # edited_cropped_image_path = out_dir / f"edited_cropped_image_from_model_{Path(image_path).stem}.png"
        # save_image(edited_cropped_image, edited_cropped_image_path)
        # print(f"Edited cropped image (output from model) saved to {edited_cropped_image_path}")

        final_image = self._paste_edited_image_back(original_image, edited_cropped_image*255.0, bbox_coords)
        
        out_path = out_dir / f"final_{Path(image_path).stem}_{target_prompt.replace(' ', '_')}.png"
        save_image(final_image / 255.0, out_path)
        
        print(f"Final edited image is saved in {out_path}")


if __name__ == "__main__":
    processor = ImageInversionProcessor()

    image_file = "./panoramic_image.jpg"
    mask_file = "./panoramic_mask1.png"
    
    
    # prompts = ["a ring","a tennis ball","a ball","a bed","a desk","a chair","a cup","a bottle","a laptop","a book"]
    prompts=[""]
    for prompt in prompts:
        processor.run_pipeline(image_file, mask_file, prompt, DoErase=True)
