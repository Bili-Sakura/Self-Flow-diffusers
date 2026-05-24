Self-Flow Diffusers integration
===============================

This repository mirrors the layout used for upstream Diffusers integration (see
[NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers.git)). Core code lives under `src/diffusers`:

- `models/transformers/transformer_selfflow.py` — `SelfFlowTransformer2DModel`
- `schedulers/scheduling_flow_match_selfflow.py` — `SelfFlowFlowMatchScheduler` (SDE flow matching)
- `pipelines/selfflow/pipeline_selfflow.py` — `SelfFlowPipeline`
- `scripts/convert_selfflow_to_diffusers.py` — checkpoint conversion

Convert a checkpoint
--------------------

```bash
python scripts/convert_selfflow_to_diffusers.py \
  --checkpoint checkpoints/selfflow_imagenet256.pt \
  --output checkpoints/selfflow-imagenet256-diffusers \
  --check-load
```

The output directory contains:

```text
model_index.json
scheduler/scheduler_config.json
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
vae_pretrained_model_name_or_path.txt
```

Sample images
-------------

```bash
python scripts/sample_selfflow.py \
  --model checkpoints/selfflow-imagenet256-diffusers \
  --class-label 207 \
  --num-inference-steps 250 \
  --guidance-scale 1.0
```

FID evaluation (50k samples)
----------------------------

```bash
python sample.py \
  --model checkpoints/selfflow-imagenet256-diffusers \
  --output-dir ./samples \
  --num-fid-samples 50000
```

Install locally
---------------

```bash
pip install -e .
```

For upstreaming to `huggingface/diffusers`, copy files under `src/diffusers` into the
matching package paths and register classes in Diffusers' lazy import tables.
