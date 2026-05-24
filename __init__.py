"""
Self-Flow ImageNet inference with native Diffusers components under `src/diffusers`.
"""

from src.diffusers._register_extensions import register_selfflow_extensions

register_selfflow_extensions()

from diffusers import SelfFlowFlowMatchScheduler, SelfFlowPipeline, SelfFlowTransformer2DModel

__all__ = [
    "SelfFlowFlowMatchScheduler",
    "SelfFlowPipeline",
    "SelfFlowTransformer2DModel",
]
