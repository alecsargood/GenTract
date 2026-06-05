import glob
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from monai import transforms
from monai.data import CacheDataset, Dataset
from sklearn.model_selection import train_test_split


def split_data(data_dir: str, test_size: float = 0.15):
    pattern = f"{data_dir}/**/*.nii.gz"
    nii_files = glob.glob(pattern, recursive=True)
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found under: {data_dir}")

    train_files, temp_files = train_test_split(nii_files, test_size=test_size * 2, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    train_data = [{"image": f, "file_path": f} for f in train_files]
    val_data = [{"image": f, "file_path": f} for f in val_files]
    test_data = [{"image": f, "file_path": f} for f in test_files]
    return train_data, val_data, test_data


def compute_global_stats(train_data, coeff: int, min_i: int, max_i: int, min_j: int, max_j: int, min_k: int, max_k: int):
    total_sum = 0.0
    total_sq_sum = 0.0
    total_voxels = 0

    for batch in train_data:
        image_path = batch["image"]
        data = nib.load(image_path).get_fdata(dtype=np.float32)
        data = data[min_i - 2:max_i + 2, min_j - 2:max_j + 2, min_k - 2:max_k + 2, coeff:coeff + 1]

        file_sum = data.sum()
        file_sq_sum = np.square(data).sum()

        total_sum += file_sum
        total_sq_sum += file_sq_sum
        total_voxels += np.prod(data.shape[:3])

    mean = total_sum / total_voxels
    variance = (total_sq_sum / total_voxels) - mean ** 2
    std = float(np.sqrt(max(variance, 1e-12)))
    return float(mean), std


def prepare_coeff_data(
    data_dir: str,
    model_type: str,
    num_unet_layers: int,
    resolution: float,
    cache: bool = False,
    augment_train: bool = True,
    coeff: int = 0,
    bbox_csv_path: Optional[str] = None,
    stats_csv_path: Optional[str] = None,
):
    if bbox_csv_path is None:
        raise ValueError("bbox_csv_path is required")

    bbox_df = pd.read_csv(bbox_csv_path, index_col="Dimension")
    min_i, max_i = int(bbox_df.loc["i", "Min Index"]), int(bbox_df.loc["i", "Max Index"])
    min_j, max_j = int(bbox_df.loc["j", "Min Index"]), int(bbox_df.loc["j", "Max Index"])
    min_k, max_k = int(bbox_df.loc["k", "Min Index"]), int(bbox_df.loc["k", "Max Index"])

    train_data, valid_data, test_data = split_data(data_dir)

    if model_type not in ["kcl", "maisi", "resnet"]:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if model_type in ["kcl", "maisi"]:
        if stats_csv_path:
            stats_df = pd.read_csv(stats_csv_path)
            train_mean = float(stats_df["means"].iloc[coeff])
            train_std = float(stats_df["stds"].iloc[coeff])
        else:
            train_mean, train_std = compute_global_stats(train_data, coeff, min_i, max_i, min_j, max_j, min_k, max_k)

        if train_std == 0:
            train_std = 1.0
        normalise_transform = transforms.NormalizeIntensityd(
            keys=["image"],
            subtrahend=train_mean,
            divisor=train_std,
        )

        select_channel = transforms.Lambdad(
            keys=["image"],
            func=lambda x: x[coeff:coeff + 1, ...],
        )
    else:
        # Kept for compatibility, but this minimal repo focuses on MAISI.
        select_channel = transforms.Lambdad(keys=["image"], func=lambda x: x)
        normalise_transform = transforms.NormalizeIntensityd(keys=["image"])

    crop_transform = transforms.SpatialCropd(
        keys=["image"],
        roi_start=(min_i - 1, min_j - 1, min_k - 1),
        roi_end=(max_i + 1, max_j + 1, max_k + 1),
    )

    loading_transforms = [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        select_channel,
        crop_transform,
        transforms.Spacingd(keys=["image"], pixdim=resolution),
        transforms.DivisiblePadd(keys=["image"], k=2 ** (num_unet_layers - 1)),
        normalise_transform,
    ]

    augmentation = [
        transforms.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1)
    ] if augment_train else []

    train_transforms = transforms.Compose(loading_transforms + augmentation)
    valid_transforms = transforms.Compose(loading_transforms)

    trainset = CacheDataset(train_data, train_transforms, cache_rate=1, num_workers=8) if cache else Dataset(train_data, train_transforms)
    validset = CacheDataset(valid_data, valid_transforms, cache_rate=1, num_workers=8) if cache else Dataset(valid_data, valid_transforms)
    testset = CacheDataset(test_data, valid_transforms, cache_rate=1, num_workers=8) if cache else Dataset(test_data, valid_transforms)

    return trainset, validset, testset
