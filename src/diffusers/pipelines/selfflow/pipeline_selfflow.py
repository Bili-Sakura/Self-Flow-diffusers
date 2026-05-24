# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

from ...models.transformers.transformer_selfflow import SelfFlowTransformer2DModel
from ...schedulers.scheduling_flow_match_selfflow import SelfFlowFlowMatchScheduler
from ...utils.token_utils import batched_prc_img, scattercat

try:
    from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
except Exception:  # pragma: no cover
    from dataclasses import dataclass

    @dataclass
    class ImagePipelineOutput:
        images: object


DEFAULT_LATENT_SIZE = 32
DEFAULT_IMAGE_SIZE = 256


class SelfFlowPipeline(DiffusionPipeline):
    """
    Pipeline for class-conditional Self-Flow image generation on ImageNet 256×256 latents.

    Default sampling uses 250 SDE steps with classifier-free guidance scale 3.5
    over flow times `(0.0, 0.7)`.
    """

    model_cpu_offload_seq = "transformer->vae"
    _optional_components = ["vae"]

    def __init__(
        self,
        transformer: SelfFlowTransformer2DModel,
        scheduler: SelfFlowFlowMatchScheduler,
        vae=None,
    ):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, vae=vae)
        self.image_processor = VaeImageProcessor()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        model_kwargs = dict(kwargs)
        transformer_subfolder = model_kwargs.pop("transformer_subfolder", None)
        scheduler_subfolder = model_kwargs.pop("scheduler_subfolder", None)
        vae_subfolder = model_kwargs.pop("vae_subfolder", None)
        base_path = Path(pretrained_model_name_or_path)

        if transformer_subfolder is None and (base_path / "transformer").exists():
            transformer_subfolder = "transformer"
        if scheduler_subfolder is None and (base_path / "scheduler").exists():
            scheduler_subfolder = "scheduler"
        if vae_subfolder is None and (base_path / "vae").exists():
            vae_subfolder = "vae"

        try:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            transformer_path = (
                str(base_path / transformer_subfolder) if transformer_subfolder else pretrained_model_name_or_path
            )
            transformer = SelfFlowTransformer2DModel.from_pretrained(transformer_path, **model_kwargs)
            try:
                scheduler = SelfFlowFlowMatchScheduler.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder=scheduler_subfolder,
                )
            except Exception:
                scheduler = SelfFlowFlowMatchScheduler()
            vae = None
            if vae_subfolder is not None:
                from diffusers import AutoencoderKL

                vae = AutoencoderKL.from_pretrained(str(base_path / vae_subfolder), **model_kwargs)
            elif (base_path / "vae_pretrained_model_name_or_path.txt").exists():
                from diffusers import AutoencoderKL

                vae_name = (base_path / "vae_pretrained_model_name_or_path.txt").read_text(encoding="utf-8").strip()
                vae = AutoencoderKL.from_pretrained(vae_name, **model_kwargs)
            return cls(transformer=transformer, scheduler=scheduler, vae=vae)

    def _patchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        patch_size = self.transformer.config.patch_size
        return rearrange(
            latents,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=patch_size,
            p2=patch_size,
        )

    def _unpatchify_latents(self, latents: torch.Tensor) -> torch.Tensor:
        patch_size = self.transformer.config.patch_size
        channels = self.transformer.config.in_channels
        return rearrange(
            latents,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=patch_size,
            p2=patch_size,
            c=channels,
        )

    def prepare_token_latents(
        self,
        batch_size: int,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_size = self.transformer.config.patch_size
        channels = self.transformer.config.in_channels
        latent_size = self.transformer.config.input_size

        latents = randn_tensor(
            (batch_size, channels, latent_size, latent_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        patched = self._patchify_latents(latents)
        tokens, token_ids = batched_prc_img(patched)
        return tokens.to(device=device, dtype=dtype), token_ids.to(device=device)

    @staticmethod
    def _apply_classifier_free_guidance(
        model_output: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        if guidance_scale <= 1.0:
            return model_output
        model_output_uncond, model_output_cond = model_output.chunk(2)
        return model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)

    def decode_latents(self, latents: torch.Tensor, output_type: str = "pil"):
        if self.vae is None:
            if output_type == "latent":
                return latents
            raise ValueError("Cannot decode latents without a VAE.")

        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        vae_dtype = next(self.vae.parameters()).dtype
        latents = (latents / scaling_factor).to(dtype=vae_dtype)
        if output_type == "latent":
            return latents
        image = self.vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)

    @torch.inference_mode()
    def __call__(
        self,
        class_labels: Union[int, List[int], torch.LongTensor],
        num_inference_steps: int = 250,
        guidance_scale: float = 3.5,
        guidance_interval: Tuple[float, float] = (0.0, 0.7),
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Generate class-conditional images with Self-Flow.

        Args:
            class_labels: ImageNet class indices.
            num_inference_steps: Number of SDE denoising steps (default `250`).
            guidance_scale: Classifier-free guidance scale (default `3.5`).
            guidance_interval: Flow-time range where CFG is applied (default `(0.0, 0.7)`).
            generator: Optional RNG for reproducibility.
            output_type: `"pil"`, `"np"`, `"pt"`, or `"latent"`.
            return_dict: Return [`ImagePipelineOutput`] if True.
        """
        device = self._execution_device
        dtype = next(self.transformer.parameters()).dtype

        if torch.is_tensor(class_labels):
            class_labels_tensor = class_labels.to(device=device, dtype=torch.long).reshape(-1)
        elif isinstance(class_labels, int):
            class_labels_tensor = torch.tensor([class_labels], device=device, dtype=torch.long)
        else:
            class_labels_tensor = torch.tensor(class_labels, device=device, dtype=torch.long).reshape(-1)

        batch_size = class_labels_tensor.shape[0]
        tokens, token_ids = self.prepare_token_latents(batch_size, generator=generator, dtype=dtype, device=device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        null_labels = torch.full_like(class_labels_tensor, self.transformer.config.num_classes - 1)
        guidance_low, guidance_high = guidance_interval

        timestep_list = self.scheduler.timesteps.tolist()
        for timestep_value in self.progress_bar(timestep_list[:-1]):
            flow_time = float(timestep_value)
            guidance_active = guidance_scale > 1.0 and guidance_low <= flow_time <= guidance_high

            if guidance_active:
                model_tokens = torch.cat([tokens, tokens], dim=0)
                labels = torch.cat([null_labels, class_labels_tensor], dim=0)
            else:
                if guidance_scale > 1.0 and tokens.shape[0] > batch_size:
                    tokens = tokens[batch_size:]
                model_tokens = tokens
                labels = class_labels_tensor

            model_t = 1.0 - flow_time
            timestep_batch = torch.full((model_tokens.shape[0],), model_t, device=device, dtype=dtype)
            model_output = self.transformer(
                model_tokens,
                timestep_batch,
                labels,
                return_dict=True,
            ).sample.to(torch.float32)
            model_output = -model_output

            if guidance_active:
                model_output = self._apply_classifier_free_guidance(model_output, guidance_scale)
                model_output = torch.cat([model_output, model_output], dim=0)

            tokens = self.scheduler.step(
                model_output,
                timestep_value,
                model_tokens,
                generator=generator,
            ).prev_sample.to(dtype)

            if guidance_active:
                tokens = tokens[batch_size:]

        if guidance_scale > 1.0 and tokens.shape[0] > batch_size:
            tokens = tokens[batch_size:]

        final_time = float(timestep_list[-1])
        final_model_t = 1.0 - final_time
        final_timestep = torch.full((tokens.shape[0],), final_model_t, device=device, dtype=dtype)
        final_output = self.transformer(
            tokens,
            final_timestep,
            class_labels_tensor,
            return_dict=True,
        ).sample.to(torch.float32)
        final_output = -final_output
        tokens = self.scheduler.step(
            final_output,
            final_time,
            tokens,
            generator=generator,
        ).prev_sample.to(dtype)

        spatial = scattercat(tokens, token_ids)
        latents = self._unpatchify_latents(spatial)
        images = self.decode_latents(latents, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)


SelfFlowPipelineOutput = ImagePipelineOutput
