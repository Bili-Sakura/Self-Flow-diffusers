#!/usr/bin/env python3
"""Sample images with a converted Self-Flow Diffusers pipeline."""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.diffusers._register_extensions import register_selfflow_extensions

register_selfflow_extensions()

from diffusers import SelfFlowPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images with SelfFlowPipeline.")
    parser.add_argument("--model", required=True, help="Path or Hub id of converted Self-Flow pipeline.")
    parser.add_argument("--class-label", type=int, action="append", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=0.7)
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="samples")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.torch_dtype]
    generator_device = args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    pipe = SelfFlowPipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    output = pipe(
        class_labels=args.class_label,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_interval=(args.guidance_low, args.guidance_high),
        generator=generator,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(output.images):
        image.save(output_dir / f"{index:06d}.png")


if __name__ == "__main__":
    main()
