import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai import transforms
from torch.amp import autocast

from Tract.models.create_models import create_maisi_vae
from Tract.utils import load_config, setup_torch


def parse_args():
    p = argparse.ArgumentParser(description="Compute per-subject stacked VAE latents")
    p.add_argument("--config", required=True, help="VAE YAML config")
    p.add_argument("--manifest-csv", required=True, help="CSV with odf_path column")
    p.add_argument("--output-dir", required=True, help="Directory for latent .npy files")
    p.add_argument("--vae-results-dir", required=True, help="Base results dir containing per-coeff VAE runs")
    p.add_argument("--bbox-csv", required=True, help="bbox.csv path")
    p.add_argument("--stats-csv", required=True, help="dists.csv path")
    p.add_argument("--odf-column", default="odf_path")
    p.add_argument("--coeff-count", type=int, default=28)
    p.add_argument("--row-index", type=int, default=None, help="Process one row index only")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--write-latent-manifest", default=None, help="Optional path to write manifest with latent_path column")
    return p.parse_args()


def output_latent_path(odf_path: str, output_dir: str):
    stem = os.path.basename(odf_path).replace(".nii.gz", "").replace(".nii", "")
    return os.path.join(output_dir, f"{stem}_latents.npy")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = setup_torch()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.manifest_csv)
    if args.odf_column not in df.columns:
        raise ValueError(f"Column '{args.odf_column}' not found in {args.manifest_csv}")

    bbox_df = pd.read_csv(args.bbox_csv, index_col="Dimension")
    min_i, max_i = int(bbox_df.loc["i", "Min Index"]), int(bbox_df.loc["i", "Max Index"])
    min_j, max_j = int(bbox_df.loc["j", "Min Index"]), int(bbox_df.loc["j", "Max Index"])
    min_k, max_k = int(bbox_df.loc["k", "Min Index"]), int(bbox_df.loc["k", "Max Index"])

    stats_df = pd.read_csv(args.stats_csv)
    means = stats_df["means"].tolist()
    stds = stats_df["stds"].tolist()

    if len(means) < args.coeff_count or len(stds) < args.coeff_count:
        raise ValueError("stats CSV has fewer rows than coeff_count")

    rows = [args.row_index] if args.row_index is not None else list(range(len(df)))
    latent_paths = [None] * len(df)

    num_unet_layers = cfg["model"].get("num_unet_layers", 4)
    resolution = cfg["data"]["resolution"]
    model_type = cfg["training"]["model"]

    for ridx in rows:
        odf_path = df.iloc[ridx][args.odf_column]
        if not isinstance(odf_path, str) or not os.path.exists(odf_path):
            raise FileNotFoundError(f"ODF file not found for row {ridx}: {odf_path}")

        latent_path = output_latent_path(odf_path, args.output_dir)
        latent_paths[ridx] = latent_path
        if args.skip_existing and os.path.exists(latent_path):
            continue

        latents = [None] * args.coeff_count

        for coeff in range(args.coeff_count):
            select_channel = transforms.Lambda(lambda data: {**data, "image": data["image"][..., coeff:coeff + 1]})
            normalise = transforms.NormalizeIntensityd(
                keys="image",
                subtrahend=means[coeff],
                divisor=stds[coeff] if stds[coeff] != 0 else 1.0,
            )
            crop = transforms.SpatialCropd(
                keys="image",
                roi_start=(min_i - 1, min_j - 1, min_k - 1),
                roi_end=(max_i + 1, max_j + 1, max_k + 1),
            )

            xforms = transforms.Compose([
                transforms.LoadImaged("image"),
                select_channel,
                transforms.EnsureChannelFirstd("image"),
                crop,
                transforms.Spacingd("image", pixdim=resolution, mode="bilinear"),
                transforms.DivisiblePadd("image", k=2 ** (num_unet_layers - 1)),
                normalise,
                transforms.EnsureTyped("image", dtype=torch.float32),
            ])

            image_batch = xforms({"image": odf_path})["image"].unsqueeze(0).to(device)

            vae = create_maisi_vae(device, cfg)
            vae_dir = os.path.join(args.vae_results_dir, f"vae_{model_type}-coeff_{coeff}")
            vae_weights = os.path.join(vae_dir, "best_autoencoder.pth")
            if not os.path.exists(vae_weights):
                raise FileNotFoundError(f"Missing VAE checkpoint: {vae_weights}")
            vae.load_state_dict(torch.load(vae_weights, map_location=device, weights_only=True))
            vae.eval()

            with torch.no_grad():
                with autocast(str(device), enabled=torch.cuda.is_available()):
                    latent = vae.encoder(image_batch)

            latents[coeff] = latent.squeeze(0).cpu().numpy()

            del vae, image_batch, latent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        stacked = np.stack(latents, axis=0)
        np.save(latent_path, stacked)

    if args.write_latent_manifest:
        out_df = df.copy()
        resolved_paths = []
        for i, row in out_df.iterrows():
            if latent_paths[i] is None:
                resolved_paths.append(output_latent_path(row[args.odf_column], args.output_dir))
            else:
                resolved_paths.append(latent_paths[i])
        out_df["latent_path"] = resolved_paths
        Path(args.write_latent_manifest).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.write_latent_manifest, index=False)


if __name__ == "__main__":
    main()
