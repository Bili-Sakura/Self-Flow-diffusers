"""Register Self-Flow modules on the installed Hugging Face `diffusers` package."""

from __future__ import annotations

from pathlib import Path


def register_selfflow_extensions() -> None:
    import diffusers.models.transformers as transformers_pkg
    import diffusers.pipelines as pipelines_pkg
    import diffusers.schedulers as schedulers_pkg
    import diffusers.utils as utils_pkg

    root = Path(__file__).resolve().parent
    extensions = (
        (transformers_pkg, root / "models" / "transformers"),
        (schedulers_pkg, root / "schedulers"),
        (pipelines_pkg, root / "pipelines"),
        (utils_pkg, root / "utils"),
    )
    for package, extension_dir in extensions:
        extension = str(extension_dir)
        if extension not in package.__path__:
            package.__path__.append(extension)

    # Re-export public API on the root diffusers module.
    import diffusers

    from diffusers.models.transformers.transformer_selfflow import (
        SelfFlowTransformer2DModel,
        SelfFlowTransformer2DModelOutput,
    )
    from diffusers.pipelines.selfflow.pipeline_selfflow import SelfFlowPipeline, SelfFlowPipelineOutput
    from diffusers.schedulers.scheduling_flow_match_selfflow import (
        SelfFlowFlowMatchScheduler,
        SelfFlowFlowMatchSchedulerOutput,
    )

    diffusers.SelfFlowTransformer2DModel = SelfFlowTransformer2DModel
    diffusers.SelfFlowTransformer2DModelOutput = SelfFlowTransformer2DModelOutput
    diffusers.SelfFlowFlowMatchScheduler = SelfFlowFlowMatchScheduler
    diffusers.SelfFlowFlowMatchSchedulerOutput = SelfFlowFlowMatchSchedulerOutput
    diffusers.SelfFlowPipeline = SelfFlowPipeline
    diffusers.SelfFlowPipelineOutput = SelfFlowPipelineOutput
