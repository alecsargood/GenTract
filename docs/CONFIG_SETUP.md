# Config And Path Setup Example

This example shows a concrete local setup with absolute paths.

## Example folder layout
```text
/data/project/
  odf/
    sub-100307_dwi_model-CSD_diffmodel.nii.gz
  tractograms/
    sub-100307_clean_tractography.trk
  tract_numpy_128/
    tracts_100307.npy
  latents/
    sub-100307_dwi_model-CSD_diffmodel_latents.npy
  manifests/
    manifest.csv
    manifest_with_latents.csv
```

## Step 1: Convert tractograms to fixed-point arrays
```bash
python scripts/preprocess_trk_to_npy.py \
  --input-root /data/project/tractograms \
  --output-dir /data/project/tract_numpy_128 \
  --pattern '*.trk' \
  --num-points 128
```

## Step 2: Build manifest
```bash
python scripts/build_manifest.py \
  --odf-root /data/project/odf \
  --tract-root /data/project/tract_numpy_128 \
  --output-csv /data/project/manifests/manifest.csv
```

## Step 3: VAE config example
Set these in `configs/maisi_vae.yaml`:
```yaml
data:
  data_dir: "/data/project/odf"
  bbox_csv: "data/examples/bbox.csv"
  stats_csv: "data/examples/dists.csv"
  resolution: 1.875
```

## Step 4: Latent extraction output into manifest
```bash
python scripts/compute_latents.py \
  --config configs/maisi_vae.yaml \
  --manifest-csv /data/project/manifests/manifest.csv \
  --output-dir /data/project/latents \
  --vae-results-dir ./results \
  --bbox-csv data/examples/bbox.csv \
  --stats-csv data/examples/dists.csv \
  --write-latent-manifest /data/project/manifests/manifest_with_latents.csv
```

## Step 5: Diffusion config example
Set this in `configs/diffusion.yaml`:
```yaml
data:
  csv_file: "/data/project/manifests/manifest_with_latents.csv"
```

## Shape requirement reminder
- Every ODF/fODF used by VAE/latents must be on the same reference grid (same shape and voxel spacing).
- For inference, first affine-register and regrid each subject ODF/fODF to that same VAE reference grid, then compute latents.
