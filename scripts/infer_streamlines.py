import argparse
import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_trk
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm

from Tract.models.create_models import create_diffusion
from Tract.utils import load_config, setup_torch


def parse_args():
    p = argparse.ArgumentParser(description="Run diffusion inference and save .trk streamlines")
    p.add_argument("--resdir", required=True, help="Results dir containing trained model and copied YAML")
    p.add_argument("--manifest-csv", required=True, help="CSV containing latent and reference image paths")
    p.add_argument("--output-dir", required=True, help="Directory to write generated .trk files")
    p.add_argument("--config", default=None, help="Optional explicit config path (defaults to first YAML in resdir)")
    p.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path (defaults to resdir/best_model.pth)")
    p.add_argument("--latent-column", default="latent_path")
    p.add_argument("--reference-column", default="odf_path")
    p.add_argument("--seed-mask-column", default=None, help="Optional CSV column with binary seed mask path")
    p.add_argument("--row-index", type=int, default=None, help="Process only one CSV row")
    p.add_argument("--inf-steps", type=int, default=50)
    p.add_argument("--num-groups", type=int, default=16)
    p.add_argument("--num-generate", type=int, default=1024, help="Streamlines per group")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def sample_and_normalize_seeds(mask_data, mask_affine, num_points, mins, maxs, device):
    voxel_coords = np.argwhere(mask_data > 0)
    if len(voxel_coords) == 0:
        raise ValueError("Seeding mask has no active voxels")

    sample_indices = np.random.choice(len(voxel_coords), size=num_points, replace=True)
    sampled_voxels = voxel_coords[sample_indices]
    jitter = np.random.uniform(-0.5, 0.5, size=sampled_voxels.shape)
    jittered_voxels = sampled_voxels.astype(np.float64) + jitter
    real_world_coords = nib.affines.apply_affine(mask_affine, jittered_voxels)
    normalized_coords = (((real_world_coords - mins) / (maxs - mins)) * 2.0) - 1.0
    return torch.from_numpy(normalized_coords).to(device=device, dtype=torch.float32)


def generate_streamlines(context, model, scheduler, device, num_generate, n_groups, seed_points=None):
    x_list = [torch.randn((num_generate, 128, 3), device=device) for _ in range(n_groups)]

    seed_chunks = None
    if seed_points is not None:
        seed_chunks = torch.chunk(seed_points, n_groups, dim=0)

    for t in tqdm(scheduler.timesteps, desc="DDIM inference", leave=False):
        with torch.no_grad():
            timestep = torch.tensor([t], device=device)
            new_x_list = []
            for i, x in enumerate(x_list):
                current_seed = seed_chunks[i] if seed_chunks is not None else None
                noise_pred = model(x=x.float(), timesteps=timestep, context=context, seed_point=current_seed)
                x_new, _ = scheduler.step(noise_pred, t, x)
                new_x_list.append(x_new)
            x_list = new_x_list

    return np.concatenate([x.detach().cpu().numpy() for x in x_list], axis=0)


def save_tractogram(streamlines, ref_nii_path, output_path):
    ref_img = nib.load(ref_nii_path)
    sft = StatefulTractogram(streamlines, ref_img, Space.RASMM)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_trk(sft, output_path, bbox_valid_check=False)


def main():
    args = parse_args()

    cfg_path = args.config or sorted(glob.glob(os.path.join(args.resdir, "*.yaml")))[0]
    cfg = load_config(cfg_path)

    checkpoint_path = args.checkpoint or os.path.join(args.resdir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = setup_torch()
    model = create_diffusion(device, cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=cfg["training"]["timesteps"],
        schedule=cfg["diffusion"]["schedule"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
    )
    scheduler.set_timesteps(num_inference_steps=args.inf_steps)

    mins = np.array(cfg["data"]["streamline_mins"], dtype=np.float32)
    maxs = np.array(cfg["data"]["streamline_maxs"], dtype=np.float32)

    df = pd.read_csv(args.manifest_csv)
    if args.latent_column not in df.columns:
        raise ValueError(f"Missing column: {args.latent_column}")
    if args.reference_column not in df.columns:
        raise ValueError(f"Missing column: {args.reference_column}")

    row_indices = [args.row_index] if args.row_index is not None else list(range(len(df)))

    for ridx in row_indices:
        row = df.iloc[ridx]
        latent_path = row[args.latent_column]
        ref_nii = row[args.reference_column]

        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Missing latent: {latent_path}")
        if not os.path.exists(ref_nii):
            raise FileNotFoundError(f"Missing reference image: {ref_nii}")

        stem = os.path.basename(latent_path).replace(".npy", "")
        out_path = os.path.join(args.output_dir, f"{stem}_if{args.inf_steps}.trk")
        if args.skip_existing and os.path.exists(out_path):
            continue

        latent = np.load(latent_path).astype(np.float32)
        context = torch.from_numpy(latent).unsqueeze(0).to(device)

        seed_points = None
        if cfg["model"].get("seed", False) and args.seed_mask_column and args.seed_mask_column in df.columns:
            seed_mask_path = row[args.seed_mask_column]
            if isinstance(seed_mask_path, str) and os.path.exists(seed_mask_path):
                seed_img = nib.load(seed_mask_path)
                seed_points = sample_and_normalize_seeds(
                    mask_data=seed_img.get_fdata(),
                    mask_affine=seed_img.affine,
                    num_points=args.num_groups * args.num_generate,
                    mins=mins,
                    maxs=maxs,
                    device=device,
                )

        generated = generate_streamlines(
            context=context,
            model=model,
            scheduler=scheduler,
            device=device,
            num_generate=args.num_generate,
            n_groups=args.num_groups,
            seed_points=seed_points,
        )

        generated = ((generated + 1.0) / 2.0) * (maxs - mins) + mins
        save_tractogram(generated, ref_nii_path=ref_nii, output_path=out_path)


if __name__ == "__main__":
    main()
