#!/usr/bin/env python3
"""Convert original Self-Flow checkpoints to a Diffusers-style pipeline directory."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.diffusers._register_extensions import register_selfflow_extensions

register_selfflow_extensions()

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except Exception:
    safe_load_file = None
    safe_save_file = None

from diffusers.models.transformers import SelfFlowTransformer2DModel
from diffusers.schedulers import SelfFlowFlowMatchScheduler


SELFFLOW_XL_CONFIG: Dict[str, Any] = {
    "input_size": 32,
    "patch_size": 2,
    "in_channels": 4,
    "hidden_size": 1152,
    "depth": 28,
    "num_heads": 16,
    "mlp_ratio": 4.0,
    "num_classes": 1001,
    "class_dropout_prob": 0.0,
    "learn_sigma": True,
    "per_token_timestep": True,
}


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "module"):
                if key in state_dict and isinstance(state_dict[key], dict):
                    state_dict = state_dict[key]
                    break
    return _clean_state_dict(state_dict)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("model.", "module.", "transformer.")
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _save_config(output_dir: Path, config: Dict[str, Any]):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(
            state_dict,
            str(output_dir / "diffusion_pytorch_model.safetensors"),
            metadata={"format": "pt"},
        )
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def _write_model_index(output_dir: Path, vae: str | None):
    model_index = {
        "_class_name": "SelfFlowPipeline",
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", "SelfFlowFlowMatchScheduler"],
        "transformer": ["diffusers", "SelfFlowTransformer2DModel"],
    }
    if vae is not None:
        model_index["vae"] = ["diffusers", "AutoencoderKL"]
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, sort_keys=True)
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Self-Flow checkpoints to Diffusers layout.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt/.bin/.safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-load", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    state_dict = _load_state_dict(args.checkpoint)
    config = {"_class_name": "SelfFlowTransformer2DModel", **SELFFLOW_XL_CONFIG}

    if args.check_load:
        model = SelfFlowTransformer2DModel(**SELFFLOW_XL_CONFIG)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            raise SystemExit(1)

    _save_config(transformer_dir, config)
    _save_weights(transformer_dir, state_dict, args.safe_serialization)
    _save_config(
        scheduler_dir,
        {
            "_class_name": "SelfFlowFlowMatchScheduler",
            "num_train_timesteps": 1000,
            "path_type": "Linear",
            "prediction": "velocity",
            "sampling_method": "Euler",
            "diffusion_form": "sigma",
            "diffusion_norm": 1.0,
            "last_step": "Mean",
            "last_step_size": 0.04,
            "reverse": True,
        },
    )
    if args.vae:
        with open(output_dir / "vae_pretrained_model_name_or_path.txt", "w", encoding="utf-8") as f:
            f.write(args.vae + os.linesep)

    _write_model_index(output_dir, args.vae)
    print(f"Saved Diffusers-style Self-Flow pipeline to {output_dir}")


if __name__ == "__main__":
    main()
