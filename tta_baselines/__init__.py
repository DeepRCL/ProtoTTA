"""Shared test-time adaptation baselines used by all four backbones."""

from .cotta import CoTTA, CoTTAImageTransform, TokenMaskTransform

__all__ = ["CoTTA", "CoTTAImageTransform", "TokenMaskTransform"]
