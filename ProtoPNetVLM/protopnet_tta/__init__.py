"""
ProtoTTA: Test-Time Adaptation for ProtoPNet.

This module provides test-time adaptation methods for ProtoPNet models,
adapted from the ProtoViT TTA framework for CNN-based prototype networks
trained on the SICAPv2 histopathology dataset.
"""

__version__ = "0.1.0"

from . import tent
from . import eata_adapt
from . import noise_utils
from . import settings
from . import preprocess

__all__ = [
    "tent",
    "eata_adapt",
    "noise_utils",
    "settings",
    "preprocess",
]
