import os
import warnings

import torch
import wandb
from torch.amp import GradScaler, autocast
from torch.nn import MSELoss
from tqdm import tqdm

from Tract.utils import AverageLoss
from Tract.utils.streamlines import apply_affine_transform_torch, generate_rotation_matrix_torch
from Tract.utils.visualizer import log_generation


def train_loop(
    model,
    diffusion_scheduler,
    optimizer,
    train_loader,
    valid_loader,
    lr_scheduler,
    device,
    cfg,
    results_dir,
):
    rotation_matrices = {}
    axes = [0, 1, 2]
    angles = [-45.0, -30.0, -15.0, 15.0, 30.0, 45.0]
    for axis in axes:
        for angle in angles:
            rotation_matrices[(axis, angle)] = generate_rotation_matrix_torch(axis, angle, device)
    rotation_matrices[(0, 0.0)] = torch.eye(4, dtype=torch.float32, device=device)

    scaler = GradScaler()
    conditioning = cfg["model"]["cond"]
    best_val_loss = float("inf")
    device_str = str(device)
    mse_loss = MSELoss()
    avgloss = AverageLoss()
    total_counter = 0
    num_timesteps = diffusion_scheduler.num_train_timesteps

    if not conditioning:
        context = None

    for epoch in range(cfg["training"]["n_epochs"]):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        progress_bar.set_description(f"Epoch {epoch}")

        for _, batch in progress_bar:
            optimizer.zero_grad()
            x = batch["tracts"].to(device).squeeze(0)
            seed_point = x[:, 0, :].clone().detach()

            if conditioning:
                context = batch["latent"].to(device)

            rot_axis = batch["rot"].item()
            deg = batch["deg"].item()
            if deg != 0.0:
                transform_matrix = rotation_matrices.get((rot_axis, deg))
                if transform_matrix is not None:
                    with torch.no_grad():
                        x = apply_affine_transform_torch(x, transform_matrix)
                else:
                    warnings.warn(f"No rotation matrix for axis={rot_axis}, deg={deg}")

            timesteps = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()

            with autocast(device_str, enabled=True):
                noise = torch.randn_like(x).to(device)
                noisy_x = diffusion_scheduler.add_noise(original_samples=x, noise=noise, timesteps=timesteps)
                noise_pred = model(x=noisy_x.float(), timesteps=timesteps, context=context, seed_point=seed_point)
                loss = mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            avgloss.put("Diffusion/mse_loss", loss.item())

            if total_counter % 10 == 0:
                avgloss.to_wandb(total_counter)
            total_counter += 1

        lr_scheduler.step()

        model.eval()
        valid_losses = []

        with torch.no_grad():
            for batch in tqdm(valid_loader, total=len(valid_loader)):
                with autocast(device_str, enabled=True):
                    x = batch["tracts"].to(device).squeeze(0)
                    seed_point = x[:, 0, :].clone().detach()

                    if conditioning:
                        context = batch["latent"].to(device)

                    rot_axis = batch["rot"].item()
                    deg = batch["deg"].item()
                    if deg != 0.0:
                        transform_matrix = rotation_matrices.get((rot_axis, deg))
                        if transform_matrix is not None:
                            x = apply_affine_transform_torch(x, transform_matrix)

                    timesteps = torch.randint(0, num_timesteps, (x.shape[0],), device=device).long()
                    noise = torch.randn_like(x).to(device)
                    noisy_x = diffusion_scheduler.add_noise(original_samples=x, noise=noise, timesteps=timesteps)
                    noise_pred = model(x=noisy_x.float(), timesteps=timesteps, context=context, seed_point=seed_point)
                    loss = mse_loss(noise_pred.float(), noise.float())
                    valid_losses.append(loss.item())

        valid_loss = sum(valid_losses) / max(len(valid_losses), 1)
        wandb.log({"Valid/mse_loss": valid_loss, "epoch": epoch + 1})

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))
            try:
                xshape = list(x.shape)
                log_generation(epoch, model, xshape, context, cfg, device, save_dir=results_dir)
            except Exception:
                pass
