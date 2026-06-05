import os

import torch
from generative.networks.nets import AutoencoderKL
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.networks.nets import PatchDiscriminator

from .cond_diff import DiffusionTransformer


def _maybe_load_weights(model: torch.nn.Module, weights_path: str, device):
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    return model


def create_kcl_vae(device, cfg):
    vae = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=3,
        num_channels=[64, 128, 128, 128],
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=[False, False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device)

    pretrained = cfg.get("vae", {}).get("pretrained", "")
    return _maybe_load_weights(vae, pretrained, device)


def create_maisi_vae(device, cfg):
    vae = AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=[64, 128, 256],
        num_res_blocks=[2, 2, 2],
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=True,
        num_splits=2,
        dim_split=1,
    ).to(device)

    pretrained = cfg.get("vae", {}).get("pretrained", "")
    return _maybe_load_weights(vae, pretrained, device)


def create_patch_discriminator(device):
    return PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(device)


def create_diffusion(device, cfg):
    model = DiffusionTransformer(**cfg["model"])
    return model.to(device)
