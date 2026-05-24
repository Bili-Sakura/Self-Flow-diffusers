#!/usr/bin/env python3
"""
Sample images from a trained Self-Flow diffusion model using the Diffusers pipeline.

Usage:
    # Convert checkpoint once
    python scripts/convert_selfflow_to_diffusers.py \\
        --checkpoint checkpoints/selfflow_imagenet256.pt \\
        --output checkpoints/selfflow-imagenet256-diffusers

    # Single GPU FID sampling
    python sample.py \\
        --model checkpoints/selfflow-imagenet256-diffusers \\
        --output-dir ./samples \\
        --num-fid-samples 50000

    # Multi-GPU
    torchrun --nnodes=1 --nproc_per_node=8 sample.py \\
        --model checkpoints/selfflow-imagenet256-diffusers \\
        --output-dir ./samples \\
        --num-fid-samples 50000
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.diffusers._register_extensions import register_selfflow_extensions

register_selfflow_extensions()

from diffusers import SelfFlowPipeline


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = 0
        torch.cuda.set_device(device)
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_npz_from_samples(samples, output_path):
    samples = np.stack(samples, axis=0)
    np.savez(output_path, arr_0=samples)
    print(f"Saved {len(samples)} samples to {output_path}")


def load_pipeline(model_path, device):
    print(f"Loading SelfFlowPipeline from {model_path}")
    pipe = SelfFlowPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    return pipe.to(device)


@torch.no_grad()
def sample_batch(pipe, batch_size, class_labels, num_steps, cfg_scale, guidance_low, guidance_high, generator):
    output = pipe(
        class_labels=class_labels,
        num_inference_steps=num_steps,
        guidance_scale=cfg_scale,
        guidance_interval=(guidance_low, guidance_high),
        generator=generator,
        output_type="np",
    )
    images = output.images
    if isinstance(images, np.ndarray) and images.ndim == 4:
        return images
    return np.stack([np.asarray(img) for img in images], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Sample images from Self-Flow (Diffusers pipeline)")
    parser.add_argument("--model", type=str, required=True, help="Path to converted Diffusers pipeline directory")
    parser.add_argument("--output-dir", type=str, default="./samples")
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=0.7)
    parser.add_argument("--save-images", action="store_true", default=True)
    parser.add_argument("--no-save-images", action="store_false", dest="save_images")
    args = parser.parse_args()

    rank, world_size, device = setup_distributed()
    device = f"cuda:{device}"

    if rank == 0:
        print(f"Running on {world_size} GPU(s)")
        print(f"Generating {args.num_fid_samples} samples")

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_images:
            (output_dir / "images").mkdir(exist_ok=True)

    if world_size > 1:
        dist.barrier()

    pipe = load_pipeline(args.model, device)

    total_samples = args.num_fid_samples
    samples_per_gpu = math.ceil(total_samples / world_size)
    start_idx = rank * samples_per_gpu
    end_idx = min(start_idx + samples_per_gpu, total_samples)
    my_samples = end_idx - start_idx

    all_samples = []
    num_batches = math.ceil(my_samples / args.batch_size)
    pbar = tqdm(range(num_batches), desc=f"GPU {rank}", disable=rank != 0)

    for batch_idx in pbar:
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, my_samples)
        batch_size = batch_end - batch_start

        class_labels = torch.randint(0, 1000, (batch_size,), device=device)
        generator = torch.Generator(device=device).manual_seed(args.seed + rank * 100000 + batch_idx)

        images = sample_batch(
            pipe,
            batch_size,
            class_labels,
            args.num_steps,
            args.cfg_scale,
            args.guidance_low,
            args.guidance_high,
            generator,
        )

        if images.dtype != np.uint8:
            images = (images * 255).clip(0, 255).astype(np.uint8)

        all_samples.append(images)

        if args.save_images and rank == 0:
            for i, img in enumerate(images):
                global_idx = start_idx + batch_start + i
                Image.fromarray(img).save(output_dir / "images" / f"{global_idx:06d}.png")

    all_samples = np.concatenate(all_samples, axis=0)

    if world_size > 1:
        rank_npz = output_dir / f"samples_rank{rank}.npz"
        np.savez(rank_npz, arr_0=all_samples)
        dist.barrier()
        if rank == 0:
            gathered = [np.load(output_dir / f"samples_rank{r}.npz")["arr_0"] for r in range(world_size)]
            for r in range(world_size):
                (output_dir / f"samples_rank{r}.npz").unlink()
            all_samples = np.concatenate(gathered, axis=0)

    if rank == 0:
        all_samples = all_samples[: args.num_fid_samples]
        npz_path = output_dir / f"samples_{len(all_samples)}.npz"
        create_npz_from_samples(list(all_samples), npz_path)
        print(f"Done! Generated {len(all_samples)} samples")
        print(f"NPZ: {npz_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
