#!/usr/bin/env python3
"""Minimal end-to-end inference from DWI inputs.

Pipeline:
1) Fit CSD SH coefficients from DWI (+ bvals/bvecs/mask) using DIPY or PyAFQ prep.
2) Build a 1-row manifest with the SH path.
3) Compute stacked VAE latents.
4) Run diffusion inference and save generated .trk streamlines.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table, unique_bvals_magnitude
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst import csdeconv as csd
from dipy.reconst import shm


def import_or_raise(module_name: str, install_hint: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"Missing dependency '{module_name}'. {install_hint}") from exc


def infer_sh_order(gtab) -> int:
    ndata = int(np.sum(~gtab.b0s_mask))
    l1 = (-3.0 + math.sqrt(1.0 + 8.0 * float(ndata))) / 2.0
    sh_order_max = int(l1)
    if sh_order_max % 2 != 0:
        sh_order_max -= 1
    return min(sh_order_max, 8)


def estimate_response(
    gtab,
    data: np.ndarray,
    response_fa_thr: float,
    b0_threshold: float,
) -> Tuple[np.ndarray, float]:
    unique_bvals = unique_bvals_magnitude(gtab.bvals)
    non_zero = unique_bvals[unique_bvals > b0_threshold]

    if len(non_zero) > 1:
        low_shell = float(non_zero[0])
        low_shell_idx = gtab.bvals <= low_shell
        response_gtab = gradient_table(
            gtab.bvals[low_shell_idx],
            bvecs=gtab.bvecs[low_shell_idx],
            b0_threshold=b0_threshold,
        )
        response_data = data[..., low_shell_idx]
    else:
        response_gtab = gtab
        response_data = data

    response, ratio = csd.auto_response_ssst(
        response_gtab,
        response_data,
        roi_radii=10,
        fa_thr=response_fa_thr,
    )
    if np.all(np.isnan(response[0])):
        raise RuntimeError("auto_response_ssst returned NaN response eigenvalues.")
    return response, float(ratio)


def fit_csd_sh(
    backend: str,
    dwi: str,
    bvals: str,
    bvecs: str,
    mask: str,
    output: str,
    b0_threshold: float,
    response_fa_thr: float,
    sh_order_max: Optional[int],
    use_afq_fix: int,
) -> None:
    if use_afq_fix == 1:
        afq_fixes = import_or_raise(
            "AFQ._fixes",
            "Install pyAFQ (pip install pyAFQ) or pass --use-afq-fix 0.",
        )
        shm.spherical_harmonics = afq_fixes.spherical_harmonics

    if backend == "pyafq":
        afq_models = import_or_raise(
            "AFQ.utils.models",
            "Install pyAFQ (pip install pyAFQ) or use --backend dipy.",
        )
        mask_img = nib.load(mask)
        mask_arr = mask_img.get_fdata(dtype=np.float32) > 0.5
        img, data, gtab, mask_data = afq_models.prepare_data(
            dwi,
            bvals,
            bvecs,
            b0_threshold=b0_threshold,
            mask=mask_arr,
        )
    else:
        bvals_arr, bvecs_arr = read_bvals_bvecs(bvals, bvecs)
        gtab = gradient_table(bvals_arr, bvecs=bvecs_arr, b0_threshold=b0_threshold)
        img = nib.load(dwi)
        data = img.get_fdata(dtype=np.float32)
        mask_img = nib.load(mask)
        mask_data = mask_img.get_fdata(dtype=np.float32) > 0.5

    final_sh_order = sh_order_max if sh_order_max is not None else infer_sh_order(gtab)
    response, _ = estimate_response(
        gtab=gtab,
        data=data,
        response_fa_thr=response_fa_thr,
        b0_threshold=b0_threshold,
    )

    csd_model = csd.ConstrainedSphericalDeconvModel(
        gtab,
        response,
        sh_order_max=final_sh_order,
    )
    csd_fit = csd_model.fit(data, mask=mask_data)
    coeff = csd_fit.shm_coeff.astype(np.float32, copy=False)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(coeff, img.affine, img.header), str(out_path))


def run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal DWI->SH->latents->diffusion inference pipeline.",
    )
    parser.add_argument("--dwi", required=True, help="Input DWI NIfTI")
    parser.add_argument("--bvals", required=True, help="Input bvals file")
    parser.add_argument("--bvecs", required=True, help="Input bvecs file")
    parser.add_argument("--mask", required=True, help="Binary brain mask NIfTI")

    parser.add_argument("--resdir", required=True, help="Diffusion results dir (contains YAML + best_model.pth)")
    parser.add_argument("--vae-results-dir", required=True, help="Base dir with VAE coeff checkpoints")
    parser.add_argument("--vae-config", required=True, help="Path to VAE YAML config")
    parser.add_argument("--bbox-csv", required=True, help="bbox.csv used for latent extraction")
    parser.add_argument("--stats-csv", required=True, help="dists.csv used for latent extraction")

    parser.add_argument("--output-dir", required=True, help="Directory for final generated .trk")
    parser.add_argument("--work-dir", required=True, help="Directory for intermediate SH/manifests/latents")
    parser.add_argument("--prefix", default=None, help="Output stem prefix (default: DWI stem)")

    parser.add_argument("--backend", choices=["pyafq", "dipy"], default="pyafq")
    parser.add_argument("--b0-threshold", type=float, default=50.0)
    parser.add_argument("--response-fa-thr", type=float, default=0.7)
    parser.add_argument("--sh-order-max", type=int, default=None)
    parser.add_argument("--use-afq-fix", type=int, choices=[0, 1], default=1)

    parser.add_argument("--inf-steps", type=int, default=50)
    parser.add_argument("--num-groups", type=int, default=16)
    parser.add_argument("--num-generate", type=int, default=1024)
    parser.add_argument("--checkpoint", default=None, help="Optional diffusion checkpoint override")
    parser.add_argument("--diff-config", default=None, help="Optional diffusion config override")
    parser.add_argument("--seed-mask", default=None, help="Optional seed mask path for seeded diffusion models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    compute_latents_script = repo_root / "scripts" / "compute_latents.py"
    infer_script = repo_root / "scripts" / "infer_streamlines.py"

    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    dwi_stem = Path(args.dwi).name
    if dwi_stem.endswith(".nii.gz"):
        dwi_stem = dwi_stem[:-7]
    elif dwi_stem.endswith(".nii"):
        dwi_stem = dwi_stem[:-4]
    prefix = args.prefix or dwi_stem

    sh_path = work_dir / f"{prefix}_sh_csd.nii.gz"
    manifest_csv = work_dir / f"{prefix}_manifest.csv"
    latent_manifest_csv = work_dir / f"{prefix}_manifest_with_latents.csv"
    latents_dir = work_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Fitting CSD SH coefficients...")
    fit_csd_sh(
        backend=args.backend,
        dwi=args.dwi,
        bvals=args.bvals,
        bvecs=args.bvecs,
        mask=args.mask,
        output=str(sh_path),
        b0_threshold=args.b0_threshold,
        response_fa_thr=args.response_fa_thr,
        sh_order_max=args.sh_order_max,
        use_afq_fix=args.use_afq_fix,
    )
    print(f"Saved SH coefficients: {sh_path}")

    print("[2/4] Writing single-row manifest...")
    manifest_fields = ["odf_path"]
    manifest_row = {"odf_path": str(sh_path)}
    if args.seed_mask:
        manifest_fields.append("seed_mask_path")
        manifest_row["seed_mask_path"] = str(Path(args.seed_mask).resolve())

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerow(manifest_row)

    print("[3/4] Computing latents...")
    run_cmd(
        [
            sys.executable,
            str(compute_latents_script),
            "--config",
            str(Path(args.vae_config).resolve()),
            "--manifest-csv",
            str(manifest_csv),
            "--output-dir",
            str(latents_dir),
            "--vae-results-dir",
            str(Path(args.vae_results_dir).resolve()),
            "--bbox-csv",
            str(Path(args.bbox_csv).resolve()),
            "--stats-csv",
            str(Path(args.stats_csv).resolve()),
            "--write-latent-manifest",
            str(latent_manifest_csv),
        ]
    )

    print("[4/4] Running diffusion inference...")
    infer_cmd = [
        sys.executable,
        str(infer_script),
        "--resdir",
        str(Path(args.resdir).resolve()),
        "--manifest-csv",
        str(latent_manifest_csv),
        "--output-dir",
        str(output_dir),
        "--row-index",
        "0",
        "--inf-steps",
        str(args.inf_steps),
        "--num-groups",
        str(args.num_groups),
        "--num-generate",
        str(args.num_generate),
    ]
    if args.checkpoint:
        infer_cmd.extend(["--checkpoint", str(Path(args.checkpoint).resolve())])
    if args.diff_config:
        infer_cmd.extend(["--config", str(Path(args.diff_config).resolve())])
    if args.seed_mask:
        infer_cmd.extend(["--seed-mask-column", "seed_mask_path"])

    run_cmd(infer_cmd)
    print(f"Done. Generated tractogram(s) in: {output_dir}")
    print(f"Intermediate files in: {work_dir}")


if __name__ == "__main__":
    main()
