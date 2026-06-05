from typing import Optional

import torch
import torch.nn as nn
from monai.networks.schedulers.ddim import DDIMScheduler
from tqdm import tqdm


@torch.no_grad()
def sample_using_diffusion(
    xshape: tuple,
    context: Optional[torch.Tensor],
    diffusion: nn.Module,
    device: str,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = "scaled_linear_beta",
    beta_start: float = 0.0015,
    beta_end: float = 0.0205,
    verbose: bool = True,
) -> torch.Tensor:
    """DDIM sampling for streamline generation."""
    scheduler = DDIMScheduler(
        num_train_timesteps=num_training_steps,
        schedule=schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        clip_sample=False,
    )
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    x = torch.randn(xshape, device=device)
    timesteps = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps

    for t in timesteps:
        timestep = torch.tensor([t], device=device)
        noise_pred = diffusion(x=x.float(), context=context, timesteps=timestep)
        x, _ = scheduler.step(noise_pred, t, x)

    return x
