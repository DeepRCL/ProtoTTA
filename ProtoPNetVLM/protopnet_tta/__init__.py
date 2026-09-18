"""
ProtoTTA: Test-Time Adaptation for ProtoPNet.

This module provides test-time adaptation methods for ProtoPNet models,
adapted from the ProtoViT TTA framework for CNN-based prototype networks
trained on the SICAPv2 histopathology dataset.
"""

__version__ = "0.1.0"

from . import tent
from . import proto_entropy
from . import eata_adapt
from . import train_and_test
from . import noise_utils
from . import settings
from . import preprocess

__all__ = [
    "tent",
    "proto_entropy",
    "eata_adapt",
    "train_and_test",
    "noise_utils",
    "settings",
    "preprocess",
]
