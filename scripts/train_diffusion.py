import argparse
import os
import shutil

import torch
import wandb
from generative.networks.schedulers import DDIMScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from Tract.data import prepare_diff_data
from Tract.models.create_models import create_diffusion
from Tract.trainer.diff_trainer import train_loop
from Tract.utils import load_config, setup_torch


def parse_args():
    p = argparse.ArgumentParser(description="Train conditional diffusion model for streamlines")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--run-name", default=None, help="Override results run name")
    p.add_argument("--csv-file", default=None, help="Override data.csv_file from config")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = setup_torch()

    csv_file = args.csv_file or cfg["data"]["csv_file"]
    num_coeffs = cfg["model"]["M"]
    num_samples = cfg["data"]["num_samples"]
    conditioning = cfg["model"]["cond"]
    seeding = cfg["model"]["seed"]

    trainset, validset, _ = prepare_diff_data(
        csv_file=csv_file,
        temp_dir=None,
        cache=cfg["data"].get("cache", False),
        num_coeffs=num_coeffs,
        num_samples=num_samples,
        cond=conditioning,
        flip=seeding,
        streamline_mins=cfg["data"].get("streamline_mins"),
        streamline_maxs=cfg["data"].get("streamline_maxs"),
    )

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 4)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    model = create_diffusion(device, cfg)
    diffusion_scheduler = DDIMScheduler(
        num_train_timesteps=cfg["training"]["timesteps"],
        schedule=cfg["diffusion"]["schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    )

    run_name = args.run_name or f"diff_{cfg['model']['model_dim']}_{cfg['model']['diff_transformer_layers']}"
    base_dir = cfg["paths"]["base_dir"]
    results_dir = os.path.join(base_dir, "results", run_name)
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(results_dir, os.path.basename(args.config)))

    wandb_mode = "online" if cfg.get("wandb", {}).get("enabled", False) else "disabled"
    wandb.init(
        project=cfg.get("wandb", {}).get("project", "gentract-open"),
        config=cfg,
        name=run_name,
        mode=wandb_mode,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["n_epochs"],
        eta_min=cfg["training"]["lr"] / 4,
    )

    train_loop(
        model=model,
        diffusion_scheduler=diffusion_scheduler,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr_scheduler=scheduler,
        device=device,
        cfg=cfg,
        results_dir=results_dir,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
