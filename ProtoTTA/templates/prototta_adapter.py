"""Copy-friendly construction template for a new prototype model.

The algorithm lives in ``ProtoTTA.adapter``. Edit only the marked
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


def add_prototta(model, prototype_class_identity, classifier_weights=None):
    """Build a ProtoTTA-wrapped version of ``model``."""

    parameters, names = configure_model(
        model,
        adapt_layer_norm=True,
        adapt_batch_norm=True,
        # Structural add-ons (Section 3.3), e.g. ("*attn*bias",) for
        # Transformers or ("*add_on_layers*weight",) for ProtoPNet-style CNNs.
        extra_parameter_patterns=(),
    )
    print("Adapting:", ", ".join(names))
    optimizer = torch.optim.Adam(
        parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0
    )

    return ProtoTTAAdapter(
        model,
        optimizer,
        prototype_class_identity=prototype_class_identity,
        classifier_weights=classifier_weights,
        output_extractor=extract_model_output,
        config=ProtoTTAConfig(
            # Choose: cosine, similarity, distance, probability, or pass
            # score_mapper=... to ProtoTTAAdapter.
            score_mapping="cosine",
            aggregation="topk_mean",
            aggregation_topk=2,
            # Threshold on the native (unmapped) prototype response.
            geometric_threshold=0.92,
            # Adaptive ProtoTTA. For the fixed-lambda variant use
            # interpolation_mode="fixed" and fixed_lambda in [0, 1].
            interpolation_mode="adaptive",
        ),
    )
