import numpy as np
import pandas as pd
import torch
from monai.config import KeysCollection
from monai.data import CacheDataset, Dataset
from monai.transforms import Compose, Lambda, LoadImaged, MapTransform, RandomizableTransform


class RandomStreamlineFlipd(RandomizableTransform, MapTransform):
    def __init__(self, keys: KeysCollection, prob: float = 0.5, allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d

        for key in self.keys:
            if key in d:
                d[key] = torch.flip(d[key], dims=[1])
        return d


class StreamlineNormalize(MapTransform):
    def __init__(self, min_coords, max_coords):
        self.min_coords = np.array(min_coords, dtype=np.float32)
        self.max_coords = np.array(max_coords, dtype=np.float32)

    def __call__(self, data):
        tracts = data["tracts"]
        if not isinstance(tracts, np.ndarray):
            tracts = np.array(tracts)
        data["tracts"] = (((tracts - self.min_coords) / (self.max_coords - self.min_coords)) * 2.0) - 1.0
        return data


class StreamlineSample(RandomizableTransform, MapTransform):
    def __init__(self, keys: KeysCollection, prob: float = 1.0, allow_missing_keys: bool = False, num_samples: int = 128) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.num_samples = num_samples

    def __call__(self, data):
        tracts = data["tracts"]
        if tracts.shape[0] < self.num_samples:
            raise ValueError(f"Cannot sample {self.num_samples} streamlines from input of shape {tracts.shape}")
        selected_inds = np.sort(np.random.choice(tracts.shape[0], size=self.num_samples, replace=False))
        data["tracts"] = tracts[selected_inds]
        return data


def split_data(csv_file: str, cond: bool = False, test: bool = False):
    df = pd.read_csv(csv_file)

    if "split" not in df.columns:
        raise ValueError("CSV must contain a 'split' column with train/val/test")

    if "rot" not in df.columns:
        df["rot"] = 0
    if "deg" not in df.columns:
        df["deg"] = 0.0

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    if test:
        if cond:
            test_data = [{"latent": r.latent_path, "odf": r.odf_path} for r in test_df.itertuples()]
        else:
            test_data = [{"tracts": r.tract_path} for r in test_df.itertuples()]
        return None, None, test_data

    if cond:
        train_data = [{"tracts": r.tract_path, "latent": r.latent_path, "rot": r.rot, "deg": r.deg} for r in train_df.itertuples()]
        val_data = [{"tracts": r.tract_path, "latent": r.latent_path, "rot": r.rot, "deg": r.deg} for r in val_df.itertuples()]
        test_data = [{"tracts": r.tract_path, "latent": r.latent_path, "rot": r.rot, "deg": r.deg} for r in test_df.itertuples()]
    else:
        train_data = [{"tracts": r.tract_path, "rot": r.rot, "deg": r.deg} for r in train_df.itertuples()]
        val_data = [{"tracts": r.tract_path, "rot": r.rot, "deg": r.deg} for r in val_df.itertuples()]
        test_data = [{"tracts": r.tract_path, "rot": r.rot, "deg": r.deg} for r in test_df.itertuples()]

    return train_data, val_data, test_data


def prepare_diff_data(
    csv_file,
    temp_dir,
    cache: bool = False,
    num_coeffs: int = 28,
    num_samples: int = 128,
    cond: bool = True,
    return_std: bool = False,
    test: bool = False,
    flip: bool = False,
    streamline_mins=None,
    streamline_maxs=None,
):
    del temp_dir, return_std

    if streamline_mins is None:
        streamline_mins = [-100.76338958740234, -115.52432250976562, -99.00054931640625]
    if streamline_maxs is None:
        streamline_maxs = [98.510498046875, 85.17140197753906, 100.56957244873047]

    train_data, valid_data, test_data = split_data(csv_file, cond=cond, test=test)

    select_coeffs = Lambda(func=lambda data: {**data, "latent": data["latent"][:num_coeffs, ...]})
    normalise_streamlines = StreamlineNormalize(streamline_mins, streamline_maxs)
    sample_streamlines = StreamlineSample(keys=["tracts"], num_samples=num_samples)

    convert_tracts = Lambda(func=lambda data: {**data, "tracts": data["tracts"].astype(np.float32)})
    convert_latent = Lambda(func=lambda data: {**data, "latent": data["latent"].astype(np.float32)})

    if test:
        test_transforms = Compose([LoadImaged(keys=["latent"]), convert_latent, select_coeffs])
        testset = CacheDataset(test_data, test_transforms, cache_rate=1, num_workers=8) if cache else Dataset(test_data, test_transforms)
        return testset

    if cond:
        loading_transforms = [
            LoadImaged(keys=["latent"]),
            convert_latent,
            select_coeffs,
            LoadImaged(keys=["tracts"]),
            normalise_streamlines,
            convert_tracts,
            sample_streamlines,
        ]
    else:
        loading_transforms = [
            LoadImaged(keys=["tracts"]),
            normalise_streamlines,
            convert_tracts,
            sample_streamlines,
        ]

    if flip:
        loading_transforms.append(RandomStreamlineFlipd(keys=["tracts"], prob=0.5))

    common_transforms = Compose(loading_transforms)

    trainset = CacheDataset(train_data, common_transforms, cache_rate=1, num_workers=8) if cache else Dataset(train_data, common_transforms)
    validset = CacheDataset(valid_data, common_transforms, cache_rate=1, num_workers=8) if cache else Dataset(valid_data, common_transforms)
    testset = CacheDataset(test_data, common_transforms, cache_rate=1, num_workers=8) if cache else Dataset(test_data, common_transforms)
    return trainset, validset, testset
