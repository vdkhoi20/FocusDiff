import json
import types
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline

from .attention import FocusDiffAttentionControl
from .config import MODEL_PRESETS, FocusDiffConfig
from .image_utils import blur_background, expand_mask, load_binary_mask, load_rgb, pil_to_model_tensor, safe_name
from .seed import seed_everything


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16}.get(name, torch.float32)


class FocusDiff:
    def __init__(
        self,
        version: str = "sd15",
        config: Optional[FocusDiffConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if version not in MODEL_PRESETS:
            raise ValueError(f"Unknown version '{version}'. Choose one of {sorted(MODEL_PRESETS)}")
        self.version = version
        self.preset = MODEL_PRESETS[version].copy()
        self.config = config or FocusDiffConfig()
        self.config.device = device or self.config.device
        self.config.height = self.preset["height"]
        self.config.width = self.preset["width"]
        if self.version == "sd21" and self.config.torch_dtype == "float16":
            bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            self.config.torch_dtype = "bfloat16" if bf16_supported else "float32"
            print(f"SD2.1 v-prediction is unstable in float16 here; using {self.config.torch_dtype}.")
        self.model_path = model_path or self.preset["model_path"]
        seed_everything(self.config.seed)
        self.model = self._load_model()

    @property
    def size(self) -> Tuple[int, int]:
        return (self.config.width, self.config.height)

    def _load_model(self):
        load_kwargs = {
            "cache_dir": self.config.cache_dir,
            "local_files_only": self.config.local_files_only,
        }
        load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}
        if self.version == "sd15":
            from .backends.sd15.diffuser_utils import FocusDiffSD15Pipeline

            scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
            return FocusDiffSD15Pipeline.from_pretrained(
                self.model_path,
                scheduler=scheduler,
                safety_checker=None,
                requires_safety_checker=False,
                **load_kwargs,
            ).to(self.config.device)

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
                **load_kwargs,
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
            **load_kwargs,
        ).to(self.config.device)
        model.scheduler = DDIMScheduler.from_config(model.scheduler.config)
        model.invert = types.MethodType(invert, model)
        model.focusdiff_call = types.MethodType(_focusdiff_call_sdxl, model)
        return model

    def _register_empty_editor(self):
        if self.version == "sd15":
            from .backends.sd15.attention_utils import AttentionBase, register_attention_editor_diffusers
        elif self.version == "sd21":
            from .backends.cpamv21.attention_utils import AttentionBase, register_attention_editor_diffusers
        else:
            from .backends.cpamvxl.attention_utils import AttentionBase, register_attention_editor_diffusers
        register_attention_editor_diffusers(self.model, AttentionBase(self.config.num_inference_steps))

    def _register_focus_editor(self, mask: torch.Tensor, do_erase: bool = False):
        if self.version == "sd15":
            from .backends.sd15.attention_utils import AttentionBase, register_attention_editor_diffusers
        elif self.version == "sd21":
            from .backends.cpamv21.attention_utils import AttentionBase, register_attention_editor_diffusers
        else:
            from .backends.cpamvxl.attention_utils import AttentionBase, register_attention_editor_diffusers

        editor = FocusDiffAttentionControl(
            AttentionBase,
            start_step=self.config.start_step,
            start_layer=self.config.start_layer,
            mask=mask,
            total_steps=self.config.num_inference_steps,
            do_erase=do_erase,
            model_type=self.preset["model_type"],
        )
        register_attention_editor_diffusers(self.model, editor)
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
        image_path: Union[str, Path],
        mask_path: Union[str, Path],
        prompt: str,
        output_path: Optional[Union[str, Path]] = None,
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
        debug_last_branch = None
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
            debug_last_branch = images[-1]
        else:
            result = self.model.focusdiff_call(
                prompt=prompts,
                latents=latents,
                ref_intermediate_objects=object_intermediates,
                ref_intermediates=source_intermediates,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                debug_output_path=output_path,
            )

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if isinstance(result, torch.Tensor):
                from torchvision.utils import save_image

                save_image(result, output_path)
                if debug_last_branch is not None:
                    debug_path = Path(output_path).with_name(f"{Path(output_path).stem}_branch_last{Path(output_path).suffix}")
                    save_image(debug_last_branch, debug_path)
            else:
                result.save(output_path)
        return result

    def run_dataset(
        self,
        root_path: Union[str, Path],
        annot_file: str = "annotates.json",
        image_dir: str = "Images",
        mask_dir: str = "Masks",
        output_dir: Union[str, Path] = "results",
        limit: Optional[int] = None,
    ):
        root = Path(root_path)
        with open(root / annot_file) as f:
            samples = json.load(f)  # type: List[dict]
        out_dir = Path(output_dir) / f"FocusDiff_{self.version}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, sample in enumerate(samples[:limit] if limit else samples):
            image_name = sample["img_name"]
            prompt = sample.get("prompt") or sample.get("target_text")
            out_path = out_dir / f"{Path(image_name).stem}_{safe_name(prompt)}.png"
            print(f"[{idx + 1}/{len(samples)}] {image_name} -> {prompt}")
            self.edit_image(root / image_dir / image_name, root / mask_dir / image_name, prompt, out_path)


@torch.no_grad()
def _focusdiff_call_sd(
    self,
    prompt,
    latents,
    ref_intermediate_objects,
    ref_intermediates,
    guidance_scale=10.0,
    num_inference_steps=50,
    debug_output_path=None,
):
    device = latents.device
    prompt_embeds, _ = self.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
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
    needs_upcasting = self.vae.dtype == torch.float16
    vae_dtype = self.vae.dtype
    if needs_upcasting:
        self.vae.to(dtype=torch.float32)
    image_latents = latents[:1].to(dtype=self.vae.dtype) / self.vae.config.scaling_factor
    image = self.vae.decode(image_latents, return_dict=False)[0]
    debug_image = None
    if debug_output_path is not None and latents.shape[0] > 3:
        debug_latents = latents[3:4].to(dtype=self.vae.dtype) / self.vae.config.scaling_factor
        debug_image = self.vae.decode(debug_latents, return_dict=False)[0]
    if needs_upcasting:
        self.vae.to(dtype=vae_dtype)
    image = self.image_processor.postprocess(image, output_type="pil")[0]
    if debug_image is not None:
        debug_image = self.image_processor.postprocess(debug_image, output_type="pil")[0]
        debug_path = Path(debug_output_path).with_name(
            f"{Path(debug_output_path).stem}_object_cfg_branch{Path(debug_output_path).suffix}"
        )
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_image.save(debug_path)
    return image


@torch.no_grad()
def _focusdiff_call_sdxl(
    self,
    prompt,
    latents,
    ref_intermediate_objects,
    ref_intermediates,
    guidance_scale=10.0,
    num_inference_steps=50,
    debug_output_path=None,
):
    device = latents.device
    height = self.default_sample_size * self.vae_scale_factor
    width = self.default_sample_size * self.vae_scale_factor
    prompt_embeds, _, pooled_prompt_embeds, _ = self.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
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
    needs_upcasting = self.vae.dtype == torch.float16 and getattr(self.vae.config, "force_upcast", False)
    if needs_upcasting:
        self.upcast_vae()
        latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != self.vae.dtype and torch.backends.mps.is_available():
        self.vae = self.vae.to(latents.dtype)

    decode_latents = latents
    image_latents = decode_latents[:1]
    has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
    has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(image_latents.device, image_latents.dtype)
        image_latents = image_latents * latents_std / self.vae.config.scaling_factor + latents_mean
    else:
        image_latents = image_latents / self.vae.config.scaling_factor

    image = self.vae.decode(image_latents, return_dict=False)[0]
    debug_image = None
    if debug_output_path is not None and decode_latents.shape[0] > 3:
        debug_latents = decode_latents[3:4]
        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(debug_latents.device, debug_latents.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(debug_latents.device, debug_latents.dtype)
            debug_latents = debug_latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            debug_latents = debug_latents / self.vae.config.scaling_factor
        debug_image = self.vae.decode(debug_latents, return_dict=False)[0]
    if needs_upcasting:
        self.vae.to(dtype=torch.float16)
    image = self.image_processor.postprocess(image, output_type="pil")[0]
    if debug_image is not None:
        debug_image = self.image_processor.postprocess(debug_image, output_type="pil")[0]
        debug_path = Path(debug_output_path).with_name(
            f"{Path(debug_output_path).stem}_object_cfg_branch{Path(debug_output_path).suffix}"
        )
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_image.save(debug_path)
    return image
