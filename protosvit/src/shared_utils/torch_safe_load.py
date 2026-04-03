import os

import torch

from src.learning.models.ClassificationModulePrune import ClassificationModulePrototype
from src.learning.models.backbones.dinov2.DinoFeaturizer import DinoFeaturizer
from src.learning.models.backbones.openclip.OpenClipFeaturizer import OpenClipFeaturizer
from src.learning.models.utils.loss import HsLoss, L1Loss, WeightedHsLoss, WeightedOrthoLoss


def register_trusted_checkpoint_globals() -> None:
    """Allow trusted local ProtoS-ViT checkpoints to load on PyTorch 2.6+."""

    # This project only loads locally produced checkpoints; disable weights-only default
    # to avoid repeatedly allowlisting deep nested backbone classes.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        return

    trusted = [
        ClassificationModulePrototype,
        DinoFeaturizer,
        OpenClipFeaturizer,
        WeightedHsLoss,
        HsLoss,
        L1Loss,
        WeightedOrthoLoss,
    ]

    # Optional, depending on installed DINOv2 package version.
    try:
        from dinov2.models.vision_transformer import DinoVisionTransformer

        trusted.append(DinoVisionTransformer)
    except Exception:
        pass

    add_safe_globals(trusted)
