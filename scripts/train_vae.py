import argparse
import os
import shutil

import torch
import wandb
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from Tract.data import prepare_coeff_data
from Tract.models.create_models import create_maisi_vae, create_patch_discriminator
from Tract.trainer.chan_vae_trainer import test_loop, train_loop
from Tract.utils import load_config, setup_torch


def parse_args():
    p = argparse.ArgumentParser(description="Train coefficient-wise MAISI VAE")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--coeff", type=int, required=True, help="SH coefficient index")
    p.add_argument("--params", type=str, default="brlp", help="Trainer weighting preset")
    p.add_argument("--data-dir", type=str, default=None, help="Override data.data_dir from config")
    p.add_argument("--bbox-csv", type=str, default=None, help="Override data.bbox_csv from config")
    p.add_argument("--stats-csv", type=str, default=None, help="Override data.stats_csv from config")
    p.add_argument("--run-name", type=str, default=None, help="Override run/results folder name")
    p.add_argument("--resume", action="store_true", help="Resume from training_checkpoint.pth")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = setup_torch()

    model_type = cfg["training"]["model"]
    run_name = args.run_name or f"vae_{model_type}-coeff_{args.coeff}"

    data_dir = args.data_dir or cfg["data"]["data_dir"]
    bbox_csv = args.bbox_csv or cfg["data"]["bbox_csv"]
    stats_csv = args.stats_csv or cfg["data"]["stats_csv"]

    base_dir = cfg["paths"]["base_dir"]
    results_dir = os.path.join(base_dir, "results", run_name)
    os.makedirs(results_dir, exist_ok=True)

    wandb_mode = "online" if cfg.get("wandb", {}).get("enabled", False) else "disabled"
    wandb.init(
        project=cfg.get("wandb", {}).get("project", "gentract-open"),
        config=cfg,
        name=run_name,
        mode=wandb_mode,
        resume="allow" if args.resume else None,
    )

    if not args.resume:
        shutil.copy(args.config, os.path.join(results_dir, os.path.basename(args.config)))

    trainset, validset, testset = prepare_coeff_data(
        model_type=model_type,
        data_dir=data_dir,
        num_unet_layers=cfg["model"].get("num_unet_layers", 4),
        resolution=cfg["data"]["resolution"],
        cache=cfg["data"].get("cache", False),
        coeff=args.coeff,
        augment_train=cfg["data"].get("augment_train", False),
        bbox_csv_path=bbox_csv,
        stats_csv_path=stats_csv,
    )

    num_workers = cfg["data"].get("num_workers", 4)
    batch_size = cfg["training"]["batch_size"]

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    autoencoder = create_maisi_vae(device, cfg)
    autoencoder.train()
    discriminator = create_patch_discriminator(device)
    discriminator.train()

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=cfg["training"]["lr"])
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["training"]["lr"])
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=cfg["training"]["n_epochs"], eta_min=cfg["training"]["lr"] / 2)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=cfg["training"]["n_epochs"], eta_min=cfg["training"]["lr"] / 2)

    scaler_g = GradScaler()
    scaler_d = GradScaler()
    start_epoch = 0
    total_counter = 0
    best_val_loss = float("inf")

    checkpoint_path = os.path.join(results_dir, "training_checkpoint.pth")
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
        scheduler_d.load_state_dict(checkpoint["scheduler_d_state_dict"])
        scaler_g.load_state_dict(checkpoint["scaler_g_state_dict"])
        scaler_d.load_state_dict(checkpoint["scaler_d_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        total_counter = checkpoint["total_counter"]
        best_val_loss = checkpoint["best_val_loss"]

    train_loop(
        autoencoder=autoencoder,
        discriminator=discriminator,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer_d=optimizer_d,
        optimizer_g=optimizer_g,
        scheduler_d=scheduler_d,
        scheduler_g=scheduler_g,
        scaler_g=scaler_g,
        scaler_d=scaler_d,
        device=device,
        cfg=cfg,
        results_dir=results_dir,
        params=args.params,
        sh_coeff=args.coeff,
        start_epoch=start_epoch,
        total_counter=total_counter,
        best_val_loss=best_val_loss,
    )

    best_autoencoder_path = os.path.join(results_dir, "best_autoencoder.pth")
    autoencoder.load_state_dict(torch.load(best_autoencoder_path, map_location=device, weights_only=True))
    test_loop(autoencoder=autoencoder, test_loader=test_loader, device=device, results_dir=results_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
