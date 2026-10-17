"""Reusable components for adding ProtoTTA to prototype-based models."""

from .adapter import (
    ProtoTTAAdapter,
    ProtoTTAConfig,
    PrototypeBatch,
    configure_model,
    default_output_extractor,
)

__all__ = [
    "ProtoTTAAdapter",
    "ProtoTTAConfig",
    "PrototypeBatch",
    "configure_model",
    "default_output_extractor",
]
