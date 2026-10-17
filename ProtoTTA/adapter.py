"""A small, model-agnostic ProtoTTA adapter.

Models can use the default dictionary output contract or provide an extractor
callback. The adapter deliberately owns no dataset or architecture logic.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


@dataclass
class PrototypeBatch:
    """Semantic outputs required for one ProtoTTA update.

    ``prototype_scores`` may have shape ``[B, P]`` or ``[B, P, K]``. The
    optional tensors make it possible to override generic routing for models
    with shared or otherwise non-standard prototypes.
    """

    logits: Tensor
    prototype_scores: Tensor
    filter_scores: Optional[Tensor] = None
    target_mask: Optional[Tensor] = None
    importance_weights: Optional[Tensor] = None


OutputExtractor = Callable[[Any], PrototypeBatch]
ScoreMapper = Callable[[Tensor], Tensor]


@dataclass
class ProtoTTAConfig:
    """Architecture-independent ProtoTTA settings.

    ``output_entropy_weight`` is the output-entropy fraction: ``0`` is pure
    ProtoTTA and values greater than zero enable the hybrid objective.
    """

    steps: int = 1
    episodic: bool = False
    score_mapping: str = "cosine"
    score_temperature: float = 1.0
    aggregation: str = "topk_mean"
    aggregation_topk: int = 1
    use_geometric_filter: bool = True
    geometric_threshold: float = 0.3
    use_confidence_weighting: bool = True
    use_importance_weighting: bool = True
    adapt_all_prototypes: bool = False
    output_entropy_weight: float = 0.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.aggregation_topk < 1:
            raise ValueError("aggregation_topk must be at least 1")
        if self.score_temperature <= 0:
            raise ValueError("score_temperature must be positive")
        if not 0.0 <= self.output_entropy_weight <= 1.0:
            raise ValueError("output_entropy_weight must be in [0, 1]")
        if not 0.0 <= self.geometric_threshold <= 1.0:
            raise ValueError("geometric_threshold must be in [0, 1]")
        if self.aggregation not in {"max", "mean", "topk_mean"}:
            raise ValueError(f"unsupported aggregation: {self.aggregation}")
        if self.score_mapping not in {
            "probability",
            "cosine",
            "similarity",
            "distance",
        }:
            raise ValueError(f"unsupported score_mapping: {self.score_mapping}")


def default_output_extractor(output: Any) -> PrototypeBatch:
    """Read the documented dictionary contract returned by a model."""

    if isinstance(output, PrototypeBatch):
        return output
    if not isinstance(output, Mapping):
        raise TypeError(
            "model output must be a mapping or PrototypeBatch; pass an "
            "output_extractor for tuple or custom outputs"
        )
    missing = {"logits", "prototype_scores"}.difference(output)
    if missing:
        raise KeyError(f"model output is missing required keys: {sorted(missing)}")
    return PrototypeBatch(
        logits=output["logits"],
        prototype_scores=output["prototype_scores"],
        filter_scores=output.get("filter_scores"),
        target_mask=output.get("target_mask"),
        importance_weights=output.get("importance_weights"),
    )


def configure_model(
    model: nn.Module,
    *,
    adapt_layer_norm: bool = True,
    adapt_batch_norm: bool = True,
    extra_parameter_patterns: Sequence[str] = (),
) -> Tuple[list[nn.Parameter], list[str]]:
    """Freeze a model and enable only the requested test-time parameters.

    BatchNorm modules stay in evaluation mode, so their running statistics are
    not modified. Glob patterns such as ``"*add_on_layers*"`` or
    ``"*prototype_vectors*"`` can opt in architecture-specific parameters.
    """

    model.eval()
    model.requires_grad_(False)
    selected: list[Tuple[str, nn.Parameter]] = []

    for module_name, module in model.named_modules():
        is_selected_norm = (
            adapt_layer_norm and isinstance(module, nn.LayerNorm)
        ) or (
            adapt_batch_norm
            and isinstance(module, nn.modules.batchnorm._BatchNorm)
        )
        if not is_selected_norm:
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name in {"weight", "bias"} and parameter is not None:
                full_name = f"{module_name}.{parameter_name}".lstrip(".")
                selected.append((full_name, parameter))

    for name, parameter in model.named_parameters():
        if any(fnmatch(name, pattern) for pattern in extra_parameter_patterns):
            selected.append((name, parameter))

    unique: list[Tuple[str, nn.Parameter]] = []
    seen = set()
    for name, parameter in selected:
        if id(parameter) in seen:
            continue
        parameter.requires_grad_(True)
        seen.add(id(parameter))
        unique.append((name, parameter))

    if not unique:
        raise ValueError(
            "no adaptation parameters were selected; enable a normalization "
            "type or provide extra_parameter_patterns"
        )
    return [parameter for _, parameter in unique], [name for name, _ in unique]


class ProtoTTAAdapter(nn.Module):
    """Wrap a prototype model with online, source-free ProtoTTA updates."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        prototype_class_identity: Optional[Tensor] = None,
        classifier_weights: Optional[Tensor] = None,
        config: Optional[ProtoTTAConfig] = None,
        output_extractor: OutputExtractor = default_output_extractor,
        score_mapper: Optional[ScoreMapper] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.config = config or ProtoTTAConfig()
        self.output_extractor = output_extractor
        self.score_mapper = score_mapper
        self.register_buffer(
            "prototype_class_identity",
            None if prototype_class_identity is None else torch.as_tensor(
                prototype_class_identity, dtype=torch.float32
            ).detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "classifier_weights",
            None if classifier_weights is None else torch.as_tensor(
                classifier_weights, dtype=torch.float32
            ).detach().clone(),
            persistent=False,
        )
        self._model_state = deepcopy(model.state_dict())
        self._optimizer_state = deepcopy(optimizer.state_dict())
        self.stats = {
            "seen_samples": 0,
            "adapted_samples": 0,
            "optimizer_steps": 0,
            "last_loss": None,
        }

    def reset(self, *, clear_stats: bool = True) -> None:
        """Restore the model and optimizer to their pre-adaptation states."""

        self.model.load_state_dict(self._model_state, strict=True)
        self.optimizer.load_state_dict(self._optimizer_state)
        if clear_stats:
            self.stats = {
                "seen_samples": 0,
                "adapted_samples": 0,
                "optimizer_steps": 0,
                "last_loss": None,
            }

    def forward_no_adapt(self, inputs: Any) -> Any:
        """Run the wrapped model without changing its parameters."""

        return self.model(inputs)

    def forward(self, inputs: Any) -> Any:
        if self.config.episodic:
            self.reset(clear_stats=False)
        output = None
        for _ in range(self.config.steps):
            output = self.forward_and_adapt(inputs)
        return output

    def _aggregate(self, scores: Tensor) -> Tensor:
        if scores.ndim == 2:
            return scores
        if scores.ndim != 3:
            raise ValueError(
                "prototype scores must have shape [B, P] or [B, P, K], "
                f"received {tuple(scores.shape)}"
            )
        if self.config.aggregation == "max":
            return scores.max(dim=2).values
        if self.config.aggregation == "mean":
            return scores.mean(dim=2)
        topk = min(self.config.aggregation_topk, scores.shape[2])
        return scores.topk(topk, dim=2).values.mean(dim=2)

    def _map_scores(self, scores: Tensor) -> Tensor:
        if self.score_mapper is not None:
            probabilities = self.score_mapper(scores)
        elif self.config.score_mapping == "probability":
            probabilities = scores
        elif self.config.score_mapping == "cosine":
            probabilities = (scores + 1.0) / 2.0
        elif self.config.score_mapping == "similarity":
            probabilities = torch.sigmoid(
                self.config.score_temperature * scores
            )
        else:
            distances = scores.clamp_min(0.0)
            probabilities = torch.log(
                (distances + 1.0) / (distances + 1e-4)
            ) / 9.0
        return probabilities.clamp(self.config.eps, 1.0 - self.config.eps)

    def _target_mask(self, batch: PrototypeBatch, scores: Tensor) -> Tensor:
        if self.config.adapt_all_prototypes:
            return torch.ones_like(scores)
        if batch.target_mask is not None:
            mask = batch.target_mask.to(device=scores.device, dtype=scores.dtype)
        else:
            predicted = batch.logits.argmax(dim=1)
            if self.prototype_class_identity is not None:
                identity = self.prototype_class_identity.to(scores.device)
                if identity.shape != (scores.shape[1], batch.logits.shape[1]):
                    raise ValueError(
                        "prototype_class_identity must have shape [P, C]; "
                        f"received {tuple(identity.shape)}"
                    )
                owners = identity.argmax(dim=1)
                mask = owners.unsqueeze(0).eq(predicted.unsqueeze(1))
            elif self.classifier_weights is not None:
                weights = self.classifier_weights.to(scores.device)
                if weights.shape != (batch.logits.shape[1], scores.shape[1]):
                    raise ValueError(
                        "classifier_weights must have shape [C, P]; "
                        f"received {tuple(weights.shape)}"
                    )
                selected = weights[predicted]
                mask = selected > 0
                empty = ~mask.any(dim=1)
                if empty.any():
                    fallback = selected.abs().argmax(dim=1)
                    mask[empty, fallback[empty]] = True
            else:
                raise ValueError("target routing information is unavailable")
            mask = mask.to(dtype=scores.dtype)
        if mask.shape != scores.shape:
            raise ValueError(
                f"target_mask must have shape {tuple(scores.shape)}, "
                f"received {tuple(mask.shape)}"
            )
        if (mask.sum(dim=1) == 0).any():
            raise ValueError("every sample must select at least one prototype")
        return mask

    def _importance(self, batch: PrototypeBatch, target_mask: Tensor) -> Tensor:
        if batch.importance_weights is not None:
            importance = batch.importance_weights.to(
                device=target_mask.device, dtype=target_mask.dtype
            ).abs()
        elif self.classifier_weights is not None:
            predicted = batch.logits.argmax(dim=1)
            importance = self.classifier_weights.to(target_mask.device)[
                predicted
            ].abs()
        else:
            importance = torch.ones_like(target_mask)
        if importance.shape != target_mask.shape:
            raise ValueError("importance_weights must have shape [B, P]")
        importance = importance * target_mask
        return importance / importance.sum(dim=1, keepdim=True).clamp_min(
            self.config.eps
        )

    @torch.enable_grad()
    def forward_and_adapt(self, inputs: Any) -> Any:
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(inputs)
        batch = self.output_extractor(output)
        scores = self._aggregate(batch.prototype_scores)
        probabilities = self._map_scores(scores)
        target_mask = self._target_mask(batch, scores)

        filter_scores = batch.filter_scores
        if filter_scores is None:
            filter_probabilities = probabilities
        else:
            filter_probabilities = self._map_scores(self._aggregate(filter_scores))
        if self.config.use_geometric_filter:
            reliable = filter_probabilities.max(dim=1).values.gt(
                self.config.geometric_threshold
            )
        else:
            reliable = torch.ones(
                batch.logits.shape[0], device=batch.logits.device, dtype=torch.bool
            )

        batch_size = int(batch.logits.shape[0])
        accepted = int(reliable.sum().item())
        self.stats["seen_samples"] += batch_size
        self.stats["adapted_samples"] += accepted
        if accepted == 0:
            return output

        binary_entropy = -(
            probabilities * probabilities.log()
            + (1.0 - probabilities) * (1.0 - probabilities).log()
        )
        if self.config.use_importance_weighting:
            importance = self._importance(batch, target_mask)
        else:
            importance = target_mask / target_mask.sum(
                dim=1, keepdim=True
            ).clamp_min(self.config.eps)
        prototype_loss_per_sample = (binary_entropy * importance).sum(dim=1)

        sample_weights = reliable.to(batch.logits.dtype)
        if self.config.use_confidence_weighting:
            sample_weights = sample_weights * batch.logits.softmax(dim=1).max(
                dim=1
            ).values.detach()
        denominator = sample_weights.sum().clamp_min(self.config.eps)
        prototype_loss = (
            prototype_loss_per_sample * sample_weights
        ).sum() / denominator

        output_probabilities = batch.logits.softmax(dim=1)
        output_entropy = -(
            output_probabilities
            * output_probabilities.clamp_min(self.config.eps).log()
        ).sum(dim=1)
        output_loss = (output_entropy * sample_weights).sum() / denominator
        output_fraction = self.config.output_entropy_weight
        loss = (1.0 - output_fraction) * prototype_loss
        loss = loss + output_fraction * output_loss
        loss.backward()
        self.optimizer.step()

        self.stats["optimizer_steps"] += 1
        self.stats["last_loss"] = float(loss.detach().item())
        return output
