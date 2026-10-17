"""Model-agnostic implementation of the submitted ProtoTTA objectives.

The adapter supports both variants reported in the paper:

* ``fixed`` uses one scalar prototype weight for the whole batch.
* ``adaptive`` computes the paper's sample-wise coefficient from activation
  margin, relative class evidence, half-batch gradient consistency, and the
  coverage guard.

Models can use the default dictionary output contract or provide an extractor
callback. Dataset and architecture details deliberately stay outside this file.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class PrototypeBatch:
    """Semantic outputs required for one ProtoTTA update.

    The first two fields are required. The optional fields override generic
    behavior for shared, multi-branch, or otherwise non-standard prototypes.
    All per-prototype tensors use shape ``[B, P]`` after aggregation.
    """

    logits: Tensor
    prototype_scores: Tensor
    filter_scores: Optional[Tensor] = None
    target_mask: Optional[Tensor] = None
    importance_weights: Optional[Tensor] = None
    prototype_targets: Optional[Tensor] = None
    reliability_mask: Optional[Tensor] = None
    controller_activations: Optional[Tensor] = None
    class_evidence: Optional[Tensor] = None
    activation_lambda: Optional[Tensor] = None


OutputExtractor = Callable[[Any], PrototypeBatch]
ScoreMapper = Callable[[Tensor], Tensor]


@dataclass
class ProtoTTAConfig:
    """Architecture-independent ProtoTTA settings.

    ``fixed_lambda`` follows the paper's convention: ``1`` is pure prototype
    entropy and ``0`` is pure output entropy. The default ``adaptive`` mode is
    the full submitted method; ``fixed`` is the reported scalar ablation.
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

    interpolation_mode: str = "adaptive"
    fixed_lambda: float = 1.0
    gradient_normalize: bool = True
    controller_topk: int = 3
    controller_delta0: float = 0.25
    controller_consistency_ratio: float = 2.0
    controller_min_consistency: float = 0.25
    controller_min_reliable: int = 4
    controller_output_floor: float = 1e-8
    use_coverage_guard: bool = True
    coverage_output_margin: float = 0.90
    coverage_max_fraction: float = 0.40
    gradient_epsilon: float = 1e-12

    entropy_epsilon: float = 1e-6
    importance_epsilon: float = 1e-6
    eps: float = 1e-8

    # Compatibility with the first public adapter. Setting this field selects
    # fixed mode and maps output fraction alpha to paper lambda = 1 - alpha.
    output_entropy_weight: Optional[float] = None

    def __post_init__(self) -> None:
        if self.output_entropy_weight is not None:
            if not 0.0 <= self.output_entropy_weight <= 1.0:
                raise ValueError("output_entropy_weight must be in [0, 1]")
            self.interpolation_mode = "fixed"
            self.fixed_lambda = 1.0 - self.output_entropy_weight
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.aggregation_topk < 1 or self.controller_topk < 1:
            raise ValueError("aggregation and controller top-k values must be positive")
        if self.controller_min_reliable < 2:
            raise ValueError("controller_min_reliable must be at least 2")
        if self.score_temperature <= 0 or self.controller_delta0 <= 0:
            raise ValueError("temperatures and controller_delta0 must be positive")
        if self.controller_consistency_ratio < 0:
            raise ValueError("controller_consistency_ratio must be non-negative")
        if (
            self.gradient_epsilon <= 0
            or self.eps <= 0
            or self.entropy_epsilon <= 0
            or self.importance_epsilon <= 0
        ):
            raise ValueError("epsilon values must be positive")
        if self.controller_output_floor < 0:
            raise ValueError("controller_output_floor must be non-negative")
        for name, value in {
            "fixed_lambda": self.fixed_lambda,
            "geometric_threshold": self.geometric_threshold,
            "controller_min_consistency": self.controller_min_consistency,
            "coverage_output_margin": self.coverage_output_margin,
            "coverage_max_fraction": self.coverage_max_fraction,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.interpolation_mode not in {"fixed", "adaptive"}:
            raise ValueError("interpolation_mode must be 'fixed' or 'adaptive'")
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
        prototype_targets=output.get("prototype_targets"),
        reliability_mask=output.get("reliability_mask"),
        controller_activations=output.get("controller_activations"),
        class_evidence=output.get("class_evidence"),
        activation_lambda=output.get("activation_lambda"),
    )


def configure_model(
    model: nn.Module,
    *,
    adapt_layer_norm: bool = True,
    adapt_batch_norm: bool = True,
    use_batch_statistics: bool = False,
    extra_parameter_patterns: Sequence[str] = (),
) -> Tuple[list[nn.Parameter], list[str]]:
    """Freeze a model and enable only the requested test-time parameters.

    BatchNorm running statistics are frozen by default. Set
    ``use_batch_statistics=True`` to match the paper's ProtoPNet protocol:
    selected BatchNorm modules use current-batch statistics without updating
    their running buffers. Glob patterns can opt in small model-specific
    modules such as add-on convolutions or attention biases.
    """

    model.eval()
    model.requires_grad_(False)
    selected: list[Tuple[str, nn.Parameter]] = []

    for module_name, module in model.named_modules():
        is_layer_norm = adapt_layer_norm and isinstance(module, nn.LayerNorm)
        is_batch_norm = adapt_batch_norm and isinstance(
            module, nn.modules.batchnorm._BatchNorm
        )
        if not (is_layer_norm or is_batch_norm):
            continue
        if is_batch_norm and use_batch_statistics:
            module.train()
            module.track_running_stats = False
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
    """Wrap a prototype model with fixed or adaptive ProtoTTA updates."""

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
        self.stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "seen_samples": 0,
            "adapted_samples": 0,
            "optimizer_steps": 0,
            "last_loss": None,
            "last_lambda_mean": None,
            "last_activation_lambda_mean": None,
            "last_relative_lambda_mean": None,
            "last_router_gate": None,
            "last_coverage": None,
            "last_proto_grad_norm": None,
            "last_output_grad_norm": None,
            "last_proto_consistency": None,
            "last_output_consistency": None,
        }

    def reset(self, *, clear_stats: bool = True) -> None:
        """Restore the model and optimizer to their pre-adaptation states."""

        self.model.load_state_dict(self._model_state, strict=True)
        self.optimizer.load_state_dict(self._optimizer_state)
        if clear_stats:
            self.stats = self._empty_stats()

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

    def _adaptation_parameters(self) -> list[nn.Parameter]:
        parameters = []
        seen = set()
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad and id(parameter) not in seen:
                    parameters.append(parameter)
                    seen.add(id(parameter))
        if not parameters:
            raise RuntimeError("optimizer has no trainable adaptation parameters")
        return parameters

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
            probabilities = (scores.clamp(-1.0, 1.0) + 1.0) / 2.0
        elif self.config.score_mapping == "similarity":
            probabilities = torch.sigmoid(
                self.config.score_temperature * scores
            )
        else:
            distances = scores.clamp_min(0.0)
            probabilities = torch.log(
                (distances + 1.0) / (distances + 1e-4)
            ) / 9.0
        if probabilities.shape != scores.shape:
            raise ValueError("score_mapper must preserve the score tensor shape")
        return probabilities.clamp(
            self.config.entropy_epsilon, 1.0 - self.config.entropy_epsilon
        )

    def _predicted_class(self, batch: PrototypeBatch) -> Tensor:
        if batch.logits.ndim != 2 or batch.logits.shape[1] < 2:
            raise ValueError("logits must have shape [B, C] with C >= 2")
        return batch.logits.argmax(dim=1)

    def _shared_prototypes(self) -> bool:
        return self.prototype_class_identity is None and self.classifier_weights is not None

    def _target_mask(self, batch: PrototypeBatch, scores: Tensor) -> Tensor:
        if self.config.adapt_all_prototypes:
            return torch.ones_like(scores)
        if batch.target_mask is not None:
            mask = batch.target_mask.to(device=scores.device, dtype=scores.dtype)
        else:
            predicted = self._predicted_class(batch)
            if self.prototype_class_identity is not None:
                identity = self.prototype_class_identity.to(scores.device)
                expected = (scores.shape[1], batch.logits.shape[1])
                if identity.shape != expected:
                    raise ValueError(
                        "prototype_class_identity must have shape [P, C]; "
                        f"expected {expected}, received {tuple(identity.shape)}"
                    )
                owners = identity.argmax(dim=1)
                mask = owners.unsqueeze(0).eq(predicted.unsqueeze(1))
            elif self.classifier_weights is not None:
                weights = self._validated_classifier_weights(batch, scores)
                mask = weights[predicted] > 0
                empty = ~mask.any(dim=1)
                if empty.any():
                    mask[empty] = True
            else:
                raise ValueError(
                    "target routing is unavailable; provide class identity, "
                    "classifier weights, or target_mask"
                )
            mask = mask.to(dtype=scores.dtype)
        if mask.shape != scores.shape:
            raise ValueError(
                f"target_mask must have shape {tuple(scores.shape)}, "
                f"received {tuple(mask.shape)}"
            )
        if (mask.sum(dim=1) == 0).any():
            raise ValueError("every sample must select at least one prototype")
        return mask

    def _validated_classifier_weights(
        self, batch: PrototypeBatch, scores: Tensor
    ) -> Tensor:
        if self.classifier_weights is None:
            raise ValueError("classifier_weights are unavailable")
        weights = self.classifier_weights.to(scores.device)
        expected = (batch.logits.shape[1], scores.shape[1])
        if weights.shape != expected:
            raise ValueError(
                "classifier_weights must have shape [C, P]; "
                f"expected {expected}, received {tuple(weights.shape)}"
            )
        return weights

    def _entropy_importance(
        self, batch: PrototypeBatch, target_mask: Tensor
    ) -> Tensor:
        if not self.config.use_importance_weighting:
            importance = torch.ones_like(target_mask)
        elif batch.importance_weights is not None:
            importance = batch.importance_weights.to(
                device=target_mask.device, dtype=target_mask.dtype
            ).abs()
        elif self.classifier_weights is not None:
            predicted = self._predicted_class(batch)
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

    def _prototype_loss_per_sample(
        self, batch: PrototypeBatch, probabilities: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return per-sample loss and controller-eligible prototype mask."""

        predicted = self._predicted_class(batch)
        shared = self._shared_prototypes() or batch.prototype_targets is not None
        if shared:
            if batch.prototype_targets is not None:
                targets = batch.prototype_targets.to(
                    device=probabilities.device, dtype=probabilities.dtype
                )
                if batch.importance_weights is None:
                    importance = torch.ones_like(probabilities)
                else:
                    importance = batch.importance_weights.to(
                        device=probabilities.device, dtype=probabilities.dtype
                    ).abs()
                if (
                    batch.target_mask is None
                    and self.prototype_class_identity is None
                    and self.classifier_weights is None
                ):
                    eligible = (targets >= 0.5).to(probabilities.dtype)
                    empty = eligible.sum(dim=1) == 0
                    if empty.any():
                        eligible[empty] = 1.0
                else:
                    eligible = self._target_mask(batch, probabilities)
            else:
                weights = self._validated_classifier_weights(batch, probabilities)
                selected = weights[predicted]
                targets = torch.sigmoid(2.0 * selected)
                importance = selected.abs()
                importance = importance / (
                    importance.max(dim=1, keepdim=True).values
                    + self.config.importance_epsilon
                )
                eligible = (selected > 0).to(probabilities.dtype)
                empty = eligible.sum(dim=1) == 0
                if empty.any():
                    eligible[empty] = 1.0
            if targets.shape != probabilities.shape or importance.shape != probabilities.shape:
                raise ValueError(
                    "prototype_targets and importance_weights must have shape [B, P]"
                )
            if not self.config.use_importance_weighting:
                importance = torch.ones_like(importance)
            bce = -(
                targets * probabilities.log()
                + (1.0 - targets) * (1.0 - probabilities).log()
            )
            return (importance * bce).mean(dim=1), eligible

        target_mask = self._target_mask(batch, probabilities)
        entropy = -(
            probabilities * probabilities.log()
            + (1.0 - probabilities) * (1.0 - probabilities).log()
        )
        importance = self._entropy_importance(batch, target_mask)
        return (entropy * importance).sum(dim=1), target_mask

    def _reliable_mask(
        self,
        batch: PrototypeBatch,
        probabilities: Tensor,
    ) -> Tensor:
        if batch.reliability_mask is not None:
            reliable = batch.reliability_mask.to(
                device=batch.logits.device, dtype=torch.bool
            )
            if reliable.shape != (batch.logits.shape[0],):
                raise ValueError("reliability_mask must have shape [B]")
            return reliable
        if not self.config.use_geometric_filter:
            return torch.ones(
                batch.logits.shape[0], device=batch.logits.device, dtype=torch.bool
            )
        if batch.filter_scores is None:
            filter_probabilities = probabilities
        else:
            filter_probabilities = self._map_scores(
                self._aggregate(batch.filter_scores)
            )
        return filter_probabilities.max(dim=1).values.gt(
            self.config.geometric_threshold
        )

    def _controller_activations(
        self, batch: PrototypeBatch, probabilities: Tensor
    ) -> Tensor:
        if batch.controller_activations is None:
            return probabilities.detach()
        activations = batch.controller_activations.to(
            device=probabilities.device, dtype=probabilities.dtype
        )
        if activations.shape != probabilities.shape:
            raise ValueError("controller_activations must have shape [B, P]")
        return activations.detach().clamp(0.0, 1.0)

    @staticmethod
    def _topk_masked_mean(values: Tensor, mask: Tensor, topk: int) -> Tensor:
        rows = []
        for row_values, row_mask in zip(values, mask.bool()):
            eligible = row_values[row_mask]
            if eligible.numel() == 0:
                eligible = row_values
            k = min(topk, int(eligible.numel()))
            rows.append(eligible.topk(k).values.mean())
        return torch.stack(rows)

    def _activation_lambda(
        self,
        batch: PrototypeBatch,
        activations: Tensor,
        eligible_mask: Tensor,
    ) -> Tensor:
        if batch.activation_lambda is not None:
            coefficient = batch.activation_lambda.to(
                device=activations.device, dtype=activations.dtype
            )
            if coefficient.shape != (activations.shape[0],):
                raise ValueError("activation_lambda must have shape [B]")
            return coefficient.detach().clamp(0.0, 1.0)
        margins = []
        for row_values, row_mask in zip(activations, eligible_mask.bool()):
            eligible = row_values[row_mask]
            if eligible.numel() == 0:
                eligible = row_values
            k = min(self.config.controller_topk, int(eligible.numel()))
            selected = eligible.topk(k).values
            margins.append((selected - 0.5).abs().mean())
        return (
            torch.stack(margins) / self.config.controller_delta0
        ).clamp(0.0, 1.0).detach()

    def _class_evidence(
        self, batch: PrototypeBatch, activations: Tensor
    ) -> Tensor:
        if batch.class_evidence is not None:
            evidence = batch.class_evidence.to(
                device=activations.device, dtype=activations.dtype
            )
            expected = (activations.shape[0], batch.logits.shape[1])
            if evidence.shape != expected:
                raise ValueError(
                    f"class_evidence must have shape {expected}, "
                    f"received {tuple(evidence.shape)}"
                )
            return evidence.detach()

        classes = batch.logits.shape[1]
        evidence = []
        if self.prototype_class_identity is not None:
            identity = self.prototype_class_identity.to(activations.device)
            expected = (activations.shape[1], classes)
            if identity.shape != expected:
                raise ValueError(
                    f"prototype_class_identity must have shape {expected}"
                )
            owners = identity.argmax(dim=1)
            for class_index in range(classes):
                mask = owners.eq(class_index).unsqueeze(0).expand_as(activations)
                if not mask.any():
                    raise ValueError(f"class {class_index} owns no prototypes")
                evidence.append(
                    self._topk_masked_mean(
                        activations, mask, self.config.controller_topk
                    )
                )
        elif self.classifier_weights is not None:
            weights = self._validated_classifier_weights(batch, activations)
            for class_index in range(classes):
                support = weights[class_index] > 0
                if not support.any():
                    k = min(self.config.controller_topk, weights.shape[1])
                    indices = weights[class_index].topk(k).indices
                    support = torch.zeros_like(support)
                    support[indices] = True
                mask = support.unsqueeze(0).expand_as(activations)
                evidence.append(
                    self._topk_masked_mean(
                        activations, mask, self.config.controller_topk
                    )
                )
        else:
            raise ValueError(
                "adaptive interpolation requires class identity, classifier "
                "weights, or class_evidence"
            )
        return torch.stack(evidence, dim=1).detach()

    def _relative_lambda(
        self, batch: PrototypeBatch, activations: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        predicted = self._predicted_class(batch)
        evidence = self._class_evidence(batch, activations)
        selected = evidence.gather(1, predicted.unsqueeze(1)).squeeze(1)
        competitor = evidence.masked_fill(
            F.one_hot(predicted, batch.logits.shape[1]).bool(), float("-inf")
        ).max(dim=1).values
        proto_margin = (
            (selected - competitor).clamp_min(0.0)
            / selected.abs().clamp_min(self.config.eps)
        ).clamp(0.0, 1.0)

        output_probabilities = batch.logits.softmax(dim=1).detach()
        top2 = output_probabilities.topk(2, dim=1).values
        output_margin = (
            (top2[:, 0] - top2[:, 1])
            / top2[:, 0].clamp_min(self.config.eps)
        ).clamp(0.0, 1.0)
        coefficient = proto_margin / (
            proto_margin + output_margin + self.config.eps
        )
        return coefficient.detach(), proto_margin.detach(), output_margin.detach()

    @staticmethod
    def _gradient_norm(gradients: Sequence[Optional[Tensor]], device: torch.device) -> Tensor:
        squared = torch.zeros((), device=device)
        for gradient in gradients:
            if gradient is not None:
                squared = squared + gradient.detach().pow(2).sum()
        return squared.sqrt()

    def _component_gradients(
        self, loss: Tensor, parameters: Sequence[nn.Parameter]
    ) -> tuple[Optional[Tensor], ...]:
        if not loss.requires_grad:
            return tuple(None for _ in parameters)
        return torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )

    def _gradient_consistency(
        self,
        loss_per_sample: Tensor,
        confidence: Tensor,
        reliable: Tensor,
        parameters: Sequence[nn.Parameter],
    ) -> Tensor:
        indices = reliable.nonzero(as_tuple=False).flatten()
        if indices.numel() < self.config.controller_min_reliable:
            return loss_per_sample.new_zeros(())
        first, second = indices[::2], indices[1::2]
        if first.numel() == 0 or second.numel() == 0:
            return loss_per_sample.new_zeros(())
        first_loss = (
            loss_per_sample[first] * confidence[first]
        ).sum() / float(first.numel())
        second_loss = (
            loss_per_sample[second] * confidence[second]
        ).sum() / float(second.numel())
        first_gradients = self._component_gradients(first_loss, parameters)
        second_gradients = self._component_gradients(second_loss, parameters)

        dot = loss_per_sample.new_zeros(())
        first_squared = loss_per_sample.new_zeros(())
        second_squared = loss_per_sample.new_zeros(())
        for first_gradient, second_gradient in zip(
            first_gradients, second_gradients
        ):
            if first_gradient is not None:
                first_squared = first_squared + first_gradient.detach().pow(2).sum()
            if second_gradient is not None:
                second_squared = second_squared + second_gradient.detach().pow(2).sum()
            if first_gradient is not None and second_gradient is not None:
                dot = dot + (
                    first_gradient.detach() * second_gradient.detach()
                ).sum()
        first_norm, second_norm = first_squared.sqrt(), second_squared.sqrt()
        if first_norm.item() == 0.0 or second_norm.item() == 0.0:
            return loss_per_sample.new_zeros(())
        cosine = (
            dot
            / (
                first_norm * second_norm
                + self.config.gradient_epsilon
            )
        ).clamp_min(0.0)
        magnitude = (
            2.0 * torch.minimum(first_norm, second_norm)
            / (
                first_norm
                + second_norm
                + self.config.gradient_epsilon
            )
        )
        return (cosine * magnitude).detach()

    def _adaptive_lambda(
        self,
        batch: PrototypeBatch,
        activations: Tensor,
        eligible_mask: Tensor,
        prototype_loss: Tensor,
        output_loss: Tensor,
        confidence: Tensor,
        reliable: Tensor,
        parameters: Sequence[nn.Parameter],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        activation = self._activation_lambda(batch, activations, eligible_mask)
        relative, proto_margin, output_margin = self._relative_lambda(
            batch, activations
        )
        proto_consistency = self._gradient_consistency(
            prototype_loss, confidence, reliable, parameters
        )
        output_consistency = self._gradient_consistency(
            output_loss, confidence, reliable, parameters
        )
        output_reference = output_consistency.clamp_min(
            self.config.controller_output_floor
        )
        consistency_gate = (
            proto_consistency
            >= self.config.controller_consistency_ratio * output_reference
        ) & (
            proto_consistency >= self.config.controller_min_consistency
        )
        coverage = reliable.float().mean()
        reliable_output_margin = output_margin[reliable].mean()
        coverage_gate = (
            self.config.use_coverage_guard
            and bool(
                (
                    reliable_output_margin
                    >= self.config.coverage_output_margin
                ).item()
            )
            and bool((coverage <= self.config.coverage_max_fraction).item())
        )
        router_gate = bool(consistency_gate.item()) or coverage_gate
        coefficient = activation if router_gate else relative
        diagnostics = {
            "activation": activation,
            "relative": relative,
            "proto_margin": proto_margin,
            "output_margin": output_margin,
            "proto_consistency": proto_consistency,
            "output_consistency": output_consistency,
            "router_gate": coefficient.new_tensor(float(router_gate)),
            "coverage": coverage,
        }
        return coefficient.detach().clamp(0.0, 1.0), diagnostics

    @torch.enable_grad()
    def forward_and_adapt(self, inputs: Any) -> Any:
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(inputs)
        batch = self.output_extractor(output)
        scores = self._aggregate(batch.prototype_scores)
        probabilities = self._map_scores(scores)
        prototype_loss, eligible_mask = self._prototype_loss_per_sample(
            batch, probabilities
        )
        reliable = self._reliable_mask(batch, probabilities)

        batch_size = int(batch.logits.shape[0])
        accepted = int(reliable.sum().item())
        self.stats["seen_samples"] += batch_size
        self.stats["adapted_samples"] += accepted
        if accepted == 0:
            return output

        output_probabilities = batch.logits.softmax(dim=1)
        output_loss = -(
            output_probabilities
            * output_probabilities.clamp_min(self.config.eps).log()
        ).sum(dim=1)
        if self.config.use_confidence_weighting:
            confidence = output_probabilities.max(dim=1).values.detach()
        else:
            confidence = torch.ones_like(output_loss)

        parameters = self._adaptation_parameters()
        reliable_weight = reliable.to(output_loss.dtype)
        denominator = reliable_weight.sum() + self.config.eps
        raw_prototype_component = (
            prototype_loss * confidence * reliable_weight
        ).sum() / denominator
        raw_output_component = (
            output_loss * confidence * reliable_weight
        ).sum() / denominator

        need_gradient_norms = self.config.gradient_normalize
        if need_gradient_norms:
            proto_gradients = self._component_gradients(
                raw_prototype_component, parameters
            )
            output_gradients = self._component_gradients(
                raw_output_component, parameters
            )
            proto_grad_norm = self._gradient_norm(
                proto_gradients, batch.logits.device
            )
            output_grad_norm = self._gradient_norm(
                output_gradients, batch.logits.device
            )
            proto_scale = proto_grad_norm.detach() + self.config.eps
            output_scale = output_grad_norm.detach() + self.config.eps
        else:
            proto_grad_norm = batch.logits.new_tensor(1.0)
            output_grad_norm = batch.logits.new_tensor(1.0)
            proto_scale = output_scale = batch.logits.new_tensor(1.0)

        diagnostics = None
        if self.config.interpolation_mode == "adaptive":
            activations = self._controller_activations(batch, probabilities)
            coefficients, diagnostics = self._adaptive_lambda(
                batch,
                activations,
                eligible_mask,
                prototype_loss,
                output_loss,
                confidence,
                reliable,
                parameters,
            )
        else:
            coefficients = torch.full_like(
                output_loss, self.config.fixed_lambda
            )

        per_sample_objective = confidence * (
            coefficients * prototype_loss / proto_scale
            + (1.0 - coefficients) * output_loss / output_scale
        )
        loss = (
            per_sample_objective * reliable_weight
        ).sum() / denominator
        loss.backward()
        self.optimizer.step()

        self.stats["optimizer_steps"] += 1
        self.stats["last_loss"] = float(loss.detach().item())
        self.stats["last_lambda_mean"] = float(
            coefficients[reliable].mean().item()
        )
        self.stats["last_proto_grad_norm"] = float(proto_grad_norm.item())
        self.stats["last_output_grad_norm"] = float(output_grad_norm.item())
        if diagnostics is not None:
            self.stats["last_activation_lambda_mean"] = float(
                diagnostics["activation"][reliable].mean().item()
            )
            self.stats["last_relative_lambda_mean"] = float(
                diagnostics["relative"][reliable].mean().item()
            )
            self.stats["last_router_gate"] = int(
                diagnostics["router_gate"].item()
            )
            self.stats["last_coverage"] = float(
                diagnostics["coverage"].item()
            )
            self.stats["last_proto_consistency"] = float(
                diagnostics["proto_consistency"].item()
            )
            self.stats["last_output_consistency"] = float(
                diagnostics["output_consistency"].item()
            )
        return output
