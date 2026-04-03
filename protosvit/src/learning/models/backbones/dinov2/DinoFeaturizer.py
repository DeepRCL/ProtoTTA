# Adapted From: https://github.com/hynnsk/HP
from typing import Callable, Optional

import torch
import torch.nn as nn

from src.learning.models.backbones.dinov2.utils import create_model


class DinoFeaturizer(nn.Module):
    """
    DinoFeaturizer is a module that extracts features from images using the DINO architecture and a projection head.

    Args:
        arch (str): The architecture name.
        dim (int): The dimension of the output features.
        proj_type (str, optional): The type of projection. Defaults to "nonlinear".
        dropout (float, optional): The dropout rate. Defaults to None.
        pretrained_weights (str, optional): The path to the pretrained weights. Defaults to None.
    """

    def __init__(
        self,
        arch: str,
        dim: int,
        proj_type: str = "nonlinear",
        dropout: float | None = None,
        pretrained_weights: Optional[str] = None,
    ):
        super().__init__()

        self.dim = dim
        self.proj_type = proj_type
        self.model = create_model(arch, pretrained_weights)
        self.patch_size = self.model.patch_size
        self.n_feats = self.model.embed_dim

        if dropout is not None:
            self.dropout = torch.nn.Dropout2d(p=dropout)
        else:
            self.dropout = torch.nn.Dropout2d(p=0)

        self.cluster1 = self.make_clusterer(self.n_feats)

        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels: int):
        return torch.nn.Sequential(torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_clusterer(self, in_channels: int):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
        )

    def forward(self, img: torch.Tensor):
        assert img.shape[2] % self.patch_size == 0
        assert img.shape[3] % self.patch_size == 0

        # get selected layer activations
        feat = self.model.forward_features(img)["x_norm_patchtokens"]
        # attention = self.model.blocks[-1].attn.attn_map
        # attention = attention[:, :, 1 + 4 :, 1 + 4 :]  # drop 1 cls token + 4 registers

        feat_h = img.shape[2] // self.patch_size
        feat_w = img.shape[3] // self.patch_size

        image_feat = feat.reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat

        return (
            self.dropout(image_feat),
            self.dropout(code),
            # attention,
        )
