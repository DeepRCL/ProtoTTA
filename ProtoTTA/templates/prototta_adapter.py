"""
Generic ProtoTTA / ProtoTTA+ implementation skeleton.

This file is intentionally lightweight and educational rather than plug-and-play.
It captures the common structure shared by the repository's ProtoViT, ProtoLens,
ProtoPFormer, ProtoS-ViT, and ProtoPNet-style implementations.

To use it for a new model, replace the TODO blocks with architecture-specific
logic for:
  - extracting logits / prototype scores / filter scores,
  - defining target prototypes,
  - selecting which parameters may adapt,
  - mapping native prototype scores into probabilities in (0, 1).
"""

from copy import deepcopy

import torch
import torch.nn as nn


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    return -(probs * torch.log(probs + 1e-6)).sum(dim=1)


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def collect_params(model, mode: str = "layernorm_only"):
    """
    Select the parameter subset allowed to adapt at test time.

    Common choices:
      - LayerNorm / BatchNorm
      - attention biases (ViT)
      - add-on conv / projection layers
      - prototype vectors
      - classification head
    """
    params, names = [], []

    for module_name, module in model.named_modules():
        if mode == "layernorm_only" and isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            for param_name, param in module.named_parameters():
                if param_name in {"weight", "bias"}:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")

    return params, names


def configure_model(model, mode: str = "layernorm_only"):
    model.train()
    model.requires_grad_(False)

    params, _ = collect_params(model, mode)
    for param in params:
        param.requires_grad_(True)

    return model


class GenericProtoTTA(nn.Module):
    """
    Skeleton implementation of ProtoTTA / ProtoTTA+.

    Set `logit_weight > 0` to activate ProtoTTA+ style hybrid loss.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        use_importance: bool = True,
        use_confidence: bool = True,
        geo_filter_threshold: float = 0.3,
        proto_weight: float = 1.0,
        logit_weight: float = 0.0,
        adapt_all_prototypes: bool = False,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.use_importance = use_importance
        self.use_confidence = use_confidence
        self.geo_filter_threshold = geo_filter_threshold
        self.proto_weight = proto_weight
        self.logit_weight = logit_weight
        self.adapt_all_prototypes = adapt_all_prototypes
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

        self.adaptation_stats = {
            "total_samples": 0,
            "adapted_samples": 0,
            "total_updates": 0,
        }

    def forward(self, x):
        self.adaptation_stats["total_samples"] += x.size(0)
        out = None
        for _ in range(self.steps):
            out = self.forward_and_adapt(x)
        return out

    def forward_no_adapt(self, x):
        return self.model(x)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def extract_outputs(self, out):
        """
        TODO:
        Return:
          logits: [B, C]
          prototype_scores: [B, P] or [B, P, K]
          filter_scores: [B, P]
        """
        raise NotImplementedError

    def aggregate_prototype_scores(self, prototype_scores: torch.Tensor) -> torch.Tensor:
        """
        Convert prototype scores to [B, P].
        Example policies:
          - return as-is if scores are already [B, P]
          - sum / max / top-k mean over sub-prototypes if [B, P, K]
        """
        if prototype_scores.dim() == 2:
            return prototype_scores
        raise NotImplementedError

    def build_target_mask(self, logits: torch.Tensor, aggregated_scores: torch.Tensor) -> torch.Tensor:
        """
        TODO:
        Return [B, P] mask identifying which prototypes should be sharpened.

        Common implementations:
          - class-specific ownership via model.prototype_class_identity
          - shared prototypes via classifier weights for the pseudo-label
        """
        raise NotImplementedError

    def compute_importance_weights(self, logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Optional architecture-specific importance weighting.
        Return [B, P] non-negative weights that sum to 1 over targets.
        """
        del logits
        eps = 1e-8
        uniform = target_mask / (target_mask.sum(dim=1, keepdim=True) + eps)
        return uniform

    def map_scores_to_probs(self, aggregated_scores: torch.Tensor) -> torch.Tensor:
        """
        TODO:
        Map native prototype scores into (0, 1).

        Examples:
          - (s + 1) / 2 for cosine similarity in [-1, 1]
          - sigmoid(temp * s)
          - distance-to-similarity conversion followed by normalization
        """
        return torch.sigmoid(aggregated_scores)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        eps = 1e-6

        out = self.model(x)
        logits, prototype_scores, filter_scores = self.extract_outputs(out)
        aggregated_scores = self.aggregate_prototype_scores(prototype_scores)

        with torch.no_grad():
            if self.geo_filter_threshold > 0:
                reliability = (filter_scores.max(dim=1)[0] > self.geo_filter_threshold).float()
            else:
                reliability = torch.ones(logits.size(0), device=logits.device)

            adapted_now = int(reliability.sum().item())
            self.adaptation_stats["adapted_samples"] += adapted_now
            if adapted_now == 0:
                return out

            if self.adapt_all_prototypes:
                target_mask = torch.ones_like(aggregated_scores)
            else:
                target_mask = self.build_target_mask(logits, aggregated_scores)

        probs = self.map_scores_to_probs(aggregated_scores).clamp(eps, 1 - eps)
        binary_entropy = -(probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs))

        sample_mask = reliability.unsqueeze(1)

        if self.use_importance:
            importance = self.compute_importance_weights(logits, target_mask)
            loss_per_sample = (binary_entropy * target_mask * importance * sample_mask).sum(dim=1)
        else:
            num_targets = target_mask.sum(dim=1).clamp(min=1.0)
            loss_per_sample = (binary_entropy * target_mask * sample_mask).sum(dim=1) / num_targets

        if self.use_confidence:
            with torch.no_grad():
                confidence = logits.softmax(dim=1).max(dim=1)[0]
            proto_loss = (loss_per_sample * confidence * reliability).sum() / (reliability.sum() + eps)
        else:
            proto_loss = (loss_per_sample * reliability).sum() / (reliability.sum() + eps)

        loss = self.proto_weight * proto_loss

        if self.logit_weight > 0:
            logit_loss = (softmax_entropy(logits) * reliability).sum() / (reliability.sum() + eps)
            loss = loss + self.logit_weight * logit_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.adaptation_stats["total_updates"] += adapted_now
        return out
