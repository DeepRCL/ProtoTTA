"""
Adapted from https://github.com/facebookresearch/dinov2
"""

import sys
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

import torch
import yaml
from omegaconf import OmegaConf

from src.learning.models.backbones.dinov2.models import vision_transformer as vits

dict_model = {
    "dinov2_vits14_reg": ["facebookresearch/dinov2", "dinov2_vits14_reg"],
    "dinov2_vitb14_reg": ["facebookresearch/dinov2", "dinov2_vitb14_reg"],
    "dinov2_vitl14_reg": ["facebookresearch/dinov2", "dinov2_vitl14_reg"],
    "dinov2_vitg14_reg": ["facebookresearch/dinov2", "dinov2_vitg14_reg"],
}


def create_model(arch: str, path_weight=None):
    """
    Create a model based on the specified architecture.

    Args:
    ----
        arch (str): The architecture name.
        path_weight (str, optional): The path to the weight file. Defaults to None.

    Returns:
    -------
        torch.nn.Module: The created model.

    """
    if arch in dict_model:
        image_encoder = torch.hub.load(*dict_model[arch])
    elif path_weight is not None:
        image_encoder, _ = create_local_model(arch, path_weight)
    else:
        raise NotImplementedError

    return image_encoder


def create_local_model(path_config: str, path_weight: str) -> tuple[Any, torch.dtype]:
    """
    Create a local model using the provided configuration file and weight file.

    Args:
    ----
        path_config (str): The path to the configuration file.
        path_weight (str): The path to the weight file.

    Returns:
    -------
        Tuple[Any, torch.dtype]: A tuple containing the created model and the autocast dtype.

    """
    cfg = OmegaConf.load(path_config)
    model, _ = build_model(
        cfg.student,
        only_teacher=True,
        img_size=cfg.crops.global_crops_size,
    )
    load_pretrained_weights(model, path_weight, checkpoint_key="teacher")
    model.cuda()
    autocast_dtype = get_autocast_dtype(cfg)
    return model, autocast_dtype


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(
            pretrained_weights,
            map_location="cpu",
        )
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        # logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)


def get_autocast_dtype(config):
    teacher_dtype_str = (
        config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    )
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float
