import argparse
import os
import random
import re
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Build GenTract manifest CSV from ODF + fixed-point tract arrays")
    p.add_argument("--odf-root", required=True, help="Root directory containing ODF .nii.gz files")
    p.add_argument("--tract-root", required=True, help="Directory containing tract numpy files")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--odf-pattern", default="*.nii.gz")
    p.add_argument("--subject-regex", default=r"sub-(\d+)")
    p.add_argument("--tract-template", default="tracts_{subject}.npy")
    p.add_argument("--latent-dir", default=None, help="Optional latent directory to add latent_path column")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def parse_rotation(name: str):
    m = re.search(r"rot(\d+)_(-?\d+(?:\.\d+)?)deg", name)
    if not m:
        return 0, 0.0
    return int(m.group(1)), float(m.group(2))


def split_subjects(subjects, train_frac, val_frac, test_frac, seed):
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    subjects = list(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_set = set(subjects[:n_train])
    val_set = set(subjects[n_train:n_train + n_val])
    test_set = set(subjects[n_train + n_val:])

    return train_set, val_set, test_set


def main():
    args = parse_args()

    odf_root = Path(args.odf_root)
    tract_root = Path(args.tract_root)
    latent_dir = Path(args.latent_dir) if args.latent_dir else None

    odf_files = sorted(odf_root.rglob(args.odf_pattern))
    if not odf_files:
        raise FileNotFoundError(f"No ODF files found with pattern '{args.odf_pattern}' under {odf_root}")

    rows = []
    subj_re = re.compile(args.subject_regex)

    for odf in odf_files:
        odf_str = str(odf)
        m = subj_re.search(odf_str)
        if not m:
            continue

        subject = m.group(1)
        tract_path = tract_root / args.tract_template.format(subject=subject)
        if not tract_path.exists():
            continue

        rot, deg = parse_rotation(odf.name)

        row = {
            "subject_id": subject,
            "odf_path": odf_str,
            "tract_path": str(tract_path),
            "rot": rot,
            "deg": deg,
        }

        if latent_dir is not None:
            latent_name = odf.name.replace(".nii.gz", "").replace(".nii", "") + "_latents.npy"
            row["latent_path"] = str(latent_dir / latent_name)

        rows.append(row)

    if not rows:
        raise RuntimeError("No matched rows were built. Check subject regex and tract template.")

    df = pd.DataFrame(rows)

    unique_subjects = sorted(df["subject_id"].unique().tolist())
    train_set, val_set, test_set = split_subjects(unique_subjects, args.train_frac, args.val_frac, args.test_frac, args.seed)

    def assign_split(sub_id):
        if sub_id in train_set:
            return "train"
        if sub_id in val_set:
            return "val"
        return "test"

    df["split"] = df["subject_id"].apply(assign_split)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
