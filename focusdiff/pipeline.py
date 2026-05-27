import json
import types
from pathlib import Path
from typing import Any

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
from pytorch_lightning import seed_everything

from .attention import FocusDiffAttentionControl
from .config import MODEL_PRESETS, FocusDiffConfig
from .image_utils import blur_background, expand_mask, load_binary_mask, load_rgb, pil_to_model_tensor, safe_name


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16}.get(name, torch.float32)


class FocusDiff:
    def __init__(
        self,
        version: str = "sd15",
        config: FocusDiffConfig | None = None,
        model_path: str | None = None,
        device: str | None = None,
    ):
        if version not in MODEL_PRESETS:
            raise ValueError(f"Unknown version '{version}'. Choose one of {sorted(MODEL_PRESETS)}")
        self.version = version
        self.preset = MODEL_PRESETS[version].copy()
        self.config = config or FocusDiffConfig()
        self.config.device = device or self.config.device
        self.config.height = self.preset["height"]
        self.config.width = self.preset["width"]
        self.model_path = model_path or self.preset["model_path"]
        seed_everything(self.config.seed)
        self.model = self._load_model()

    @property
    def size(self) -> tuple[int, int]:
        return (self.config.width, self.config.height)

    def _load_model(self):
        if self.version == "sd15":
            from .backends.sd15.diffuser_utils import OIICtrlPipeline

            scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
            return OIICtrlPipeline.from_pretrained(self.model_path, scheduler=scheduler).to(self.config.device)

        if self.version == "sd21":
            from .backends.cpamv21.diffuser_utils import invert

            scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                prediction_type="v_prediction",
                steps_offset=1,
            )
            model = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                scheduler=scheduler,
                torch_dtype=_dtype(self.config.torch_dtype),
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.config.device)
            model.invert = types.MethodType(invert, model)
            model.focusdiff_call = types.MethodType(_focusdiff_call_sd, model)
            return model

        from .backends.cpamvxl.diffuser_utils import invert

        model = StableDiffusionXLPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.config.torch_dtype == "float16" else _dtype(self.config.torch_dtype),
            variant="fp16" if self.config.torch_dtype == "float16" else None,
            use_safetensors=True,
        ).to(self.config.device)
        model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
        model.invert = types.MethodType(invert, model)
        model.focusdiff_call = types.MethodType(_focusdiff_call_sdxl, model)
        return model

    def _register_empty_editor(self):
        if self.version == "sd15":
            from .backends.sd15.OIIctrl_utils import AttentionBase, regiter_attention_editor_diffusers
        elif self.version == "sd21":
            from .backends.cpamv21.OIIctrl_utils import AttentionBase, regiter_attention_editor_diffusers
        else:
            from .backends.cpamvxl.OIIctrl_utils import AttentionBase, regiter_attention_editor_diffusers
        regiter_attention_editor_diffusers(self.model, AttentionBase(self.config.num_inference_steps))

    def _register_focus_editor(self, mask: torch.Tensor, do_erase: bool = False):
        if self.version == "sd15":
            from .backends.sd15.OIIctrl_utils import AttentionBase, regiter_attention_editor_diffusers
        elif self.version == "sd21":
            from .backends.cpamv21.OIIctrl_utils import AttentionBase, regiter_attention_editor_diffusers
        else:
            from .backends.cpamvxl.OIIctrl_utils import AttentionBase, regiter_attention_editor_diffusers

        editor = FocusDiffAttentionControl(
            AttentionBase,
            start_step=self.config.start_step,
            start_layer=self.config.start_layer,
            mask=mask,
            total_steps=self.config.num_inference_steps,
            do_erase=do_erase,
            model_type=self.preset["model_type"],
        )
        regiter_attention_editor_diffusers(self.model, editor)
        return editor

    def _invert(self, image):
        self._register_empty_editor()
        if self.version == "sd15":
            tensor = pil_to_model_tensor(image, self.config.device)
            return self.model.invert(
                tensor,
                prompt="",
                guidance_scale=0,
                num_inference_steps=self.config.num_inference_steps,
                return_intermediates=True,
                DEVICE=self.config.device,
            )
        if self.version == "sd21":
            intermediates, start_code = self.model.invert(
                "",
                image,
                guidance_scale=0.0,
                eta=0.0,
                num_inference_steps=self.config.num_inference_steps,
            )
            return start_code, intermediates
        return self.model.invert(
            image,
            prompt="",
            guidance_scale=1,
            num_inference_steps=self.config.num_inference_steps,
        )

    def edit_image(
        self,
        image_path: str | Path,
        mask_path: str | Path,
        prompt: str,
        output_path: str | Path | None = None,
        do_erase: bool = False,
    ):
        image = load_rgb(image_path, self.size)
        mask = load_binary_mask(mask_path, self.config.device, self.size)
        focused_mask = expand_mask(mask, self.config.mask_scale).to(self.config.device)
        blurred = blur_background(image, mask, self.config.blur_kernel, self.config.blur_sigma)

        start_object, object_intermediates = self._invert(blurred)
        start_source, source_intermediates = self._invert(image)
        latents = torch.cat([start_source.clone(), start_source.clone(), start_object.clone(), start_object.clone()])

        self._register_focus_editor(focused_mask, do_erase=do_erase)
        prompts = ["", "", "", "", prompt]
        if self.version == "sd15":
            images = self.model(
                prompts,
                latents=latents,
                ref_intermediate_objects=object_intermediates,
                ref_intermediates=source_intermediates,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                DEVICE=self.config.device,
            )
            result = images[0]
        else:
            result = self.model.focusdiff_call(
                prompt=prompts,
                latents=latents,
                ref_intermediate_objects=object_intermediates,
                ref_intermediates=source_intermediates,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
            )

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if isinstance(result, torch.Tensor):
                from torchvision.utils import save_image

                save_image(result, output_path)
            else:
                result.save(output_path)
        return result

    def run_dataset(
        self,
        root_path: str | Path,
        annot_file: str = "annotates.json",
        image_dir: str = "Images",
        mask_dir: str = "Masks",
        output_dir: str | Path = "results",
        limit: int | None = None,
    ):
        root = Path(root_path)
        with open(root / annot_file) as f:
            samples: list[dict[str, Any]] = json.load(f)
        out_dir = Path(output_dir) / f"FocusDiff_{self.version}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, sample in enumerate(samples[:limit] if limit else samples):
            image_name = sample["img_name"]
            prompt = sample.get("prompt") or sample.get("target_text")
            out_path = out_dir / f"{Path(image_name).stem}_{safe_name(prompt)}.png"
            print(f"[{idx + 1}/{len(samples)}] {image_name} -> {prompt}")
            self.edit_image(root / image_dir / image_name, root / mask_dir / image_name, prompt, out_path)


@torch.no_grad()
def _focusdiff_call_sd(self, prompt, latents, ref_intermediate_objects, ref_intermediates, guidance_scale=10.0, num_inference_steps=50):
    device = self._execution_device
    prompt_embeds, _ = self.encode_prompt(prompt, device, 1, False)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents.to(device=device, dtype=prompt_embeds.dtype) * self.scheduler.init_noise_sigma
    for i, t in enumerate(self.scheduler.timesteps):
        model_inputs = latents.clone()
        model_inputs[1] = ref_intermediates[-1 - i].to(device=device, dtype=latents.dtype)
        model_inputs[2] = ref_intermediate_objects[-1 - i].to(device=device, dtype=latents.dtype)
        model_inputs = torch.cat([model_inputs, latents[-1:].clone()], dim=0)
        model_inputs = self.scheduler.scale_model_input(model_inputs, t)
        noise = self.unet(model_inputs, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        noise_uncond, noise_text = noise[-2], noise[-1]
        noise_target = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        noise = torch.cat([noise[:-2], noise_target[None]], dim=0)
        latents = self.scheduler.step(noise, t, latents, return_dict=False)[0]
    image = self.vae.decode(latents[:1] / self.vae.config.scaling_factor, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type="pil")[0]
    return image


@torch.no_grad()
def _focusdiff_call_sdxl(self, prompt, latents, ref_intermediate_objects, ref_intermediates, guidance_scale=10.0, num_inference_steps=50):
    device = self._execution_device
    height = self.default_sample_size * self.vae_scale_factor
    width = self.default_sample_size * self.vae_scale_factor
    prompt_embeds, _, pooled_prompt_embeds, _ = self.encode_prompt(prompt, device, 1, False)
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents.to(device=device, dtype=prompt_embeds.dtype) * self.scheduler.init_noise_sigma
    add_time_ids = self._get_add_time_ids(
        (height, width),
        (0, 0),
        (height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=int(pooled_prompt_embeds.shape[-1]),
    ).to(device)
    add_time_ids = add_time_ids.repeat(len(prompt), 1)
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds.to(device), "time_ids": add_time_ids}
    for i, t in enumerate(self.scheduler.timesteps):
        model_inputs = latents.clone()
        model_inputs[1] = ref_intermediates[-1 - i].to(device=device, dtype=latents.dtype)
        model_inputs[2] = ref_intermediate_objects[-1 - i].to(device=device, dtype=latents.dtype)
        model_inputs = torch.cat([model_inputs, latents[-1:].clone()], dim=0)
        model_inputs = self.scheduler.scale_model_input(model_inputs, t)
        noise = self.unet(
            model_inputs,
            t,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        noise_uncond, noise_text = noise[-2], noise[-1]
        noise_target = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        noise = torch.cat([noise[:-2], noise_target[None]], dim=0)
        latents = self.scheduler.step(noise, t, latents, return_dict=False)[0]
    image = self.vae.decode(latents[:1] / self.vae.config.scaling_factor, return_dict=False)[0]
    return self.image_processor.postprocess(image, output_type="pil")[0]
