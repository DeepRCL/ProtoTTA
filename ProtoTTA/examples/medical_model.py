"""Minimal class-specific prototype model with a ProtoTTA wrapper.

Run from the repository root with:
    python -m ProtoTTA.examples.medical_model

The model is a toy stand-in; replace the encoder and prototypes with a real
model. Reliable-set thresholds are model-specific (Appendix L), so the value
below only makes this toy example adapt on random inputs.
"""

import torch
from torch import nn

from ProtoTTA import ProtoTTAAdapter, ProtoTTAConfig, configure_model


class MedicalPrototypeModel(nn.Module):
    """Small CNN with cosine-similarity prototypes and class identities."""

    def __init__(self, num_classes: int = 2, prototypes_per_class: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        num_prototypes = num_classes * prototypes_per_class
        self.prototype_vectors = nn.Parameter(torch.randn(num_prototypes, 8))
        owners = torch.arange(num_classes).repeat_interleave(prototypes_per_class)
        self.register_buffer(
            "prototype_class_identity",
            torch.nn.functional.one_hot(owners, num_classes).float(),
        )

    def forward(self, images):
        features = self.encoder(images).flatten(1)
        features = torch.nn.functional.normalize(features, dim=1)
        prototypes = torch.nn.functional.normalize(self.prototype_vectors, dim=1)
        similarities = features @ prototypes.t()
        logits = similarities @ self.prototype_class_identity
        return {
            "logits": logits,
            "prototype_scores": similarities,
            "filter_scores": similarities,
        }


def build_adapted_model(model: MedicalPrototypeModel) -> ProtoTTAAdapter:
    # Normalization affine parameters, with current-batch BatchNorm statistics.
    parameters, names = configure_model(
        model,
        adapt_layer_norm=False,
        adapt_batch_norm=True,
        use_batch_statistics=True,
    )
    print("Adapting:", ", ".join(names))
    optimizer = torch.optim.Adam(
        parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0
    )
    return ProtoTTAAdapter(
        model,
        optimizer,
        prototype_class_identity=model.prototype_class_identity,
        config=ProtoTTAConfig(
            score_mapping="cosine",
            geometric_threshold=0.0,  # raw cosine; choose per model
            interpolation_mode="adaptive",
        ),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    adapted_model = build_adapted_model(MedicalPrototypeModel())
    predictions = adapted_model(torch.randn(8, 1, 64, 64))["logits"].argmax(1)
    print("Predictions:", predictions.tolist())
    print("Adaptation stats:", adapted_model.stats)
