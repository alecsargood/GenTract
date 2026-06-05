import yaml
from typing import Union

import wandb

from .torch_utils import setup_torch
from .visualizer import save_nifti_image, plot_reconstruction
from .streamlines import generate_rotation_matrix_torch, apply_affine_transform_torch


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AverageLoss:
    """Track scalar metrics and periodically flush their means."""

    def __init__(self):
        self.losses_accumulator = {}

    def put(self, loss_key: str, loss_value: Union[int, float]) -> None:
        if loss_key not in self.losses_accumulator:
            self.losses_accumulator[loss_key] = []
        self.losses_accumulator[loss_key].append(loss_value)

    def pop_avg(self, loss_key: str):
        if loss_key not in self.losses_accumulator or not self.losses_accumulator[loss_key]:
            return None
        losses = self.losses_accumulator[loss_key]
        self.losses_accumulator[loss_key] = []
        return sum(losses) / len(losses)

    def get_avg(self, loss_key: str):
        if loss_key not in self.losses_accumulator or not self.losses_accumulator[loss_key]:
            return None
        losses = self.losses_accumulator[loss_key]
        return sum(losses) / len(losses)

    def to_wandb(self, step: int):
        for metric_key in list(self.losses_accumulator.keys()):
            val = self.pop_avg(metric_key)
            if val is not None:
                wandb.log({metric_key: val, "iter": step})
