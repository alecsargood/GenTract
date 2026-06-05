import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import wandb

from .sampling import sample_using_diffusion


def save_nifti_image(data: np.ndarray, save_path: str):
    nii_img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii_img, save_path)


def plot_reconstruction(original: np.ndarray, generated: np.ndarray, save_dir: str, filename: str = "recon.png"):
    depth = original.shape[-1]
    slice_indices = [int(np.round(depth / 4)), int(np.round(depth / 2)), int(np.round(depth * 3 / 4))]

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    for i, slice_idx in enumerate(slice_indices):
        axes[i, 0].imshow(original[:, :, slice_idx], cmap="gray")
        axes[i, 0].set_title(f"Original {slice_idx}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(generated[:, :, slice_idx], cmap="gray")
        axes[i, 1].set_title(f"Reconstruction {slice_idx}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close(fig)


def log_reconstruction(step: int, image, recon, coeff: int, save_dir: str):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(7, 5))
    for _ax in ax.flatten():
        _ax.set_axis_off()

    if len(image.shape) == 4:
        image = image.squeeze(0)
    if len(recon.shape) == 4:
        recon = recon.squeeze(0)

    ax[0, 0].set_title(f"original c_{coeff}", color="magenta")
    ax[0, 0].imshow(image[image.shape[0] // 2, :, :], cmap="gray")
    ax[0, 1].imshow(image[:, image.shape[1] // 2, :], cmap="gray")
    ax[0, 2].imshow(image[:, :, image.shape[2] // 2], cmap="gray")

    ax[1, 0].set_title(f"recon c_{coeff}", color="cyan")
    ax[1, 0].imshow(recon[recon.shape[0] // 2, :, :], cmap="gray")
    ax[1, 1].imshow(recon[:, recon.shape[1] // 2, :], cmap="gray")
    ax[1, 2].imshow(recon[:, :, recon.shape[2] // 2], cmap="gray")

    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_recon.png")
    plt.savefig(save_path)
    wandb.log({"plot": wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_generation(epoch, diffusion, xshape, context, cfg, device, save_dir, wandb_name: str = "generation"):
    diffusion.eval()
    tracts = sample_using_diffusion(
        xshape=xshape,
        context=context,
        diffusion=diffusion,
        device=device,
        num_training_steps=cfg["training"]["timesteps"],
        num_inference_steps=cfg["diffusion"]["num_inference_steps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    )
    tracts = tracts.detach().cpu().numpy()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Generated streamlines")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for tract in tracts:
        ax.plot(tract[:, 0], tract[:, 1], tract[:, 2], color="red", alpha=0.5)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_{wandb_name}.png")
    plt.savefig(save_path)
    wandb.log({f"Plots/{wandb_name}": wandb.Image(fig), "epoch": epoch})
    plt.close(fig)
