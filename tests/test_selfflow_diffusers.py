import pytest

torch = pytest.importorskip("torch")

from diffusers.models.transformers.transformer_selfflow import SelfFlowTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_selfflow import SelfFlowFlowMatchScheduler


def test_transformer_forward_tokens():
    model = SelfFlowTransformer2DModel(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=11,
        learn_sigma=False,
        per_token_timestep=True,
    )
    tokens = torch.randn(2, 16, 16)
    timesteps = torch.tensor([0.5, 0.25])
    class_labels = torch.tensor([1, 2])
    output = model(tokens, timesteps, class_labels)
    assert output.sample.shape == tokens.shape


def test_scheduler_sde_step_runs():
    scheduler = SelfFlowFlowMatchScheduler()
    scheduler.set_timesteps(4)
    sample = torch.randn(2, 16, 16)
    velocity = torch.zeros_like(sample)
    output = scheduler.step(velocity, scheduler.timesteps[0], sample)
    assert output.prev_sample.shape == sample.shape
