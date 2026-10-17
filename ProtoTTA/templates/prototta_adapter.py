"""Copy-friendly construction template for a new prototype model.

The reusable algorithm lives in ``ProtoTTA.adapter``. Edit only the marked
model-specific values here instead of copying the adaptation algorithm.
"""

import torch

from ProtoTTA import (
    ProtoTTAAdapter,
    ProtoTTAConfig,
    PrototypeBatch,
    configure_model,
)


def extract_model_output(output):
    """Adapt this function if the model does not return the default dict."""

    # Example tuple contract. Replace indices/names for the target model.
    logits, prototype_scores = output[:2]
    return PrototypeBatch(
        logits=logits,
        prototype_scores=prototype_scores,
        filter_scores=prototype_scores,
    )


def add_prototta(model, prototype_class_identity):
    """Build a ProtoTTA-wrapped version of ``model``."""

    parameters, names = configure_model(
        model,
        adapt_layer_norm=True,
        adapt_batch_norm=True,
        # Add model-specific parameters only if normalization is insufficient.
        extra_parameter_patterns=(),
    )
    if not names:
        raise RuntimeError("No parameters selected for adaptation")
    optimizer = torch.optim.Adam(parameters, lr=1e-4)

    return ProtoTTAAdapter(
        model,
        optimizer,
        prototype_class_identity=prototype_class_identity,
        output_extractor=extract_model_output,
        config=ProtoTTAConfig(
            # Choose: cosine, similarity, distance, or probability.
            score_mapping="cosine",
            aggregation="topk_mean",
            aggregation_topk=1,
            geometric_threshold=0.3,
            # Full paper method. For the scalar ablation, use
            # interpolation_mode="fixed" and set fixed_lambda in [0, 1].
            interpolation_mode="adaptive",
            gradient_normalize=True,
        ),
    )
