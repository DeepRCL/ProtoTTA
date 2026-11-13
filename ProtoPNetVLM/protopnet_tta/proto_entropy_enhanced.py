"""
Prototype-aware entropy minimization for ProtoPNet.

ProtoEntropyEnhanced blends the binary entropy of the predicted-class prototype
similarities with output entropy using a fixed weight (mode ProtoHybrid).
SamplewiseAdaptiveProtoEntropy sets the weight per sample with the label-free
controller (router, single estimates, mean of the two).
"""

import torch
import torch.nn as nn
import math


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class ProtoEntropyEnhanced(nn.Module):
    """Prototype-aware entropy minimization with a fixed prototype weight.

    The loss blends the binary entropy of the predicted-class prototype
    similarities (weight alpha_proto) with the output entropy (weight
    alpha_softmax) over the samples that pass the entropy and geometric
    reliability filters.
    """

    def __init__(self, model, optimizer, steps=1,
                 # Loss weights
                 alpha_proto=1.0, alpha_softmax=0.0,
                 # Filtering
                 entropy_margin_scale=0.4,
                 geo_filter_threshold=0.3):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "ProtoEntropyEnhanced requires >= 1 step(s)"

        # Loss weights
        self.alpha_proto = alpha_proto  # Weight for binary prototype entropy
        self.alpha_softmax = alpha_softmax  # Weight for softmax entropy
        # Subclasses that calculate a per-sample lambda set this to True.
        # The fixed-lambda path uses it only to emit a compressed audit record;
        # it does not participate in the loss or gradient computation.
        self.samplewise_adaptive_lambda = False

        # Filtering settings
        self.entropy_margin_scale = entropy_margin_scale  # 0.4 * log(num_classes)
        self.geo_filter_threshold = geo_filter_threshold

        # Tracking
        self.batch_count = 0
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'filtered_by_entropy': 0,
            'filtered_by_geo': 0,
            'total_updates': 0,
        }
        self.selected_prototype_weight_sum = 0.0
        self.selected_prototype_weight_count = 0
        self.geo_filter_stats = {
            'total_samples': 0,
            'filtered_samples': 0,
            'min_similarities': [],
            'max_similarities': [],
            'avg_similarities': []
        }

    def forward(self, x):
        batch_size = x.size(0)
        self.adaptation_stats['total_samples'] += batch_size

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)

        self.batch_count += 1

        return outputs

    def _get_model_components(self):
        """Get prototype_class_identity and last_layer from model."""
        if hasattr(self.model, 'core'):
            proto_class_identity = self.model.core.prototype_class_identity
            last_layer = self.model.core.last_layer
        else:
            proto_class_identity = self.model.prototype_class_identity
            last_layer = self.model.last_layer
        return proto_class_identity, last_layer

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Single forward pass and one update on the reliable samples."""
        outputs = self.model(x)

        if isinstance(outputs, tuple) and len(outputs) >= 2:
            logits, min_distances = outputs[0], outputs[1]
        else:
            logits = outputs
            min_distances = None

        if min_distances is None:
            # Fallback to simple softmax entropy
            loss = softmax_entropy(logits).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            return outputs

        loss, reliable_mask = self._compute_loss(logits, min_distances)

        if loss.item() > 0 and reliable_mask.sum() > 0:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Track adapted samples
            self.adaptation_stats['adapted_samples'] += int(reliable_mask.sum().item())
            self.adaptation_stats['total_updates'] += 1

        return outputs

    def _compute_loss(self, logits, min_distances):
        """Compute the fixed-weight loss with filtering.

        Returns:
            loss: Combined loss value
            reliable_mask: Mask of reliable samples
        """
        device = logits.device
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        # Convert distances to similarities
        raw_similarities = torch.log((min_distances + 1.0) / (min_distances + 1e-4))  # [B, P]
        similarities = raw_similarities / 9.0  # Normalize to roughly [0, 1]

        # Get prototype class identity
        proto_class_identity, last_layer = self._get_model_components()

        with torch.no_grad():
            pred_class = logits.argmax(dim=1)  # [B]
            proto_identities = proto_class_identity.argmax(dim=1).to(device)  # [P]

        # ============ FILTERING ============
        reliable_mask = torch.ones(batch_size, device=device)

        # 1. Entropy filtering
        margin_e0 = self.entropy_margin_scale * math.log(num_classes)
        entropy = softmax_entropy(logits)
        entropy_mask = (entropy < margin_e0).float()
        reliable_mask = reliable_mask * entropy_mask
        self.adaptation_stats['filtered_by_entropy'] += int((1 - entropy_mask).sum().item())

        # 2. Geometric filtering
        max_sim_per_sample = similarities.max(dim=1)[0]  # [B]
        geo_mask = (max_sim_per_sample > self.geo_filter_threshold).float()

        reliable_mask = reliable_mask * geo_mask
        self.adaptation_stats['filtered_by_geo'] += int((1 - geo_mask).sum().item())

        # Track geo stats
        self.geo_filter_stats['total_samples'] += batch_size
        self.geo_filter_stats['filtered_samples'] += int((1 - geo_mask).sum().item())

        if reliable_mask.sum() == 0:
            return torch.tensor(0.0, device=device), reliable_mask

        sample_weights = reliable_mask.unsqueeze(1)  # [B, 1]

        # ============ LOSS COMPUTATION ============

        # Predicted-class prototypes
        target_mask = (proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)).float()

        # Diagnostic only: mean absolute classifier weight of the predicted-class
        # prototypes selected for reliable samples. This does not affect the loss.
        with torch.no_grad():
            reliable = reliable_mask.bool()
            if reliable.any():
                selected_weights = torch.abs(last_layer.weight[pred_class]) * target_mask
                mean_weight_per_sample = (
                    selected_weights.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)
                )
                self.selected_prototype_weight_sum += float(
                    mean_weight_per_sample[reliable].sum().item()
                )
                self.selected_prototype_weight_count += int(reliable.sum().item())

        loss = torch.tensor(0.0, device=device)

        # ========== Part A: Binary Prototype Entropy ==========
        if self.alpha_proto > 0:
            eps = 1e-6
            masked_sims = similarities * target_mask
            masked_sims = torch.clamp(masked_sims, min=0.0, max=1.0)
            proto_probs = torch.clamp(masked_sims, min=eps, max=1-eps)

            # Binary entropy
            entropy = -(proto_probs * torch.log(proto_probs) +
                       (1 - proto_probs) * torch.log(1 - proto_probs))

            # Uniform mean over the predicted-class prototypes
            masked_entropy = entropy * target_mask * sample_weights
            loss_per_sample = masked_entropy.sum(dim=1) / (target_mask.sum(dim=1) + 1e-8)

            loss_proto = (loss_per_sample * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)

            loss = loss + self.alpha_proto * loss_proto

        # ========== Part B: Softmax Entropy ==========
        if self.alpha_softmax > 0:
            softmax_ent = softmax_entropy(logits)  # [B]
            loss_softmax = (softmax_ent * reliable_mask).sum() / (reliable_mask.sum() + 1e-8)
            loss = loss + self.alpha_softmax * loss_softmax

        return loss, reliable_mask

    def get_stats(self):
        """Get adaptation statistics."""
        mean_selected_weight = None
        if self.selected_prototype_weight_count > 0:
            mean_selected_weight = (
                self.selected_prototype_weight_sum /
                self.selected_prototype_weight_count
            )
        stats = {
            **self.adaptation_stats,
            'mean_selected_prototype_weight': mean_selected_weight,
            'selected_prototype_weight_count': self.selected_prototype_weight_count,
            'selected_prototype_weight_definition': (
                'mean absolute last-layer weight over predicted-class prototypes '
                'for reliable samples'
            ),
            'geo_filter_stats': self.geo_filter_stats.copy()
        }
        if not self.samplewise_adaptive_lambda:
            accepted_count = int(self.adaptation_stats['adapted_samples'])
            stats['fixed_lambda_audit'] = {
                'representation': (
                    'one constant scalar plus its accepted-sample count; '
                    'equivalent to recording the same scalar once per sample'
                ),
                'prototype_lambda': float(self.alpha_proto),
                'output_lambda': float(self.alpha_softmax),
                'accepted_sample_count': accepted_count,
                'recorded_lambda_count': accepted_count,
                'distinct_prototype_lambda_values': [float(self.alpha_proto)],
                'constant_for_every_accepted_sample': True,
            }
        return stats


class SamplewiseAdaptiveProtoEntropy(ProtoEntropyEnhanced):
    """ProtoTTA+ with a detached, per-sample prototype-loss weight."""

    def __init__(self, *args, adaptive_delta0=0.25, adaptive_top_k=3,
                 component_gradient_normalization=False,
                 adaptive_controller='absolute_consistency_router', **kwargs):
        super().__init__(*args, **kwargs)
        self.samplewise_adaptive_lambda = True
        if adaptive_delta0 <= 0:
            raise ValueError('adaptive_delta0 must be positive')
        if adaptive_top_k <= 0:
            raise ValueError('adaptive_top_k must be positive')
        if adaptive_controller not in {
                'absolute_consistency_router', 'absolute_only_router',
                'forced_relative', 'forced_native', 'mean_act_rel'}:
            raise ValueError(
                'adaptive_controller must be absolute_consistency_router, '
                'absolute_only_router, forced_relative, forced_native, or '
                'mean_act_rel'
            )
        self.adaptive_delta0 = adaptive_delta0
        self.adaptive_top_k = adaptive_top_k
        self.adaptive_controller = adaptive_controller
        self.component_gradient_normalization = component_gradient_normalization
        self.lambda_values = []
        self.proto_loss_sum = 0.0
        self.output_loss_sum = 0.0
        self.component_loss_sample_count = 0
        self.proto_component_sum = 0.0
        self.output_component_sum = 0.0
        self.component_batch_count = 0
        self.proto_gradient_norms = []
        self.output_gradient_norms = []
        self.proto_margin_values = []
        self.output_margin_values = []
        self.proto_output_agreement_values = []
        self._last_controller_diagnostics = None
        self.coverage_router_native_batches = 0
        self.coverage_router_total_batches = 0
        self.coverage_router_q_proto_values = []
        self.coverage_router_q_out_values = []
        self.coverage_router_reliable_fractions = []
        self.coverage_router_gate_native = 0
        self.coverage_router_coverage_native = 0
        self.routing_diagnostic_values = {
            'q_proto': [],
            'q_out': [],
            'q_proto_over_q_out': [],
            'native_lambda': [],
            'relative_lambda': [],
            'selected_lambda': [],
            'reliable_fraction': [],
            'native_path_fraction': [],
            'accepted_samples': [],
        }
        self.prototype_score_values = []

    def _optimizer_parameters(self):
        return [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group['params']
            if parameter.requires_grad
        ]

    def _gradient_norm(self, loss):
        parameters = self._optimizer_parameters()
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm = torch.zeros((), device=loss.device)
        for gradient in gradients:
            if gradient is not None:
                squared_norm = squared_norm + gradient.detach().pow(2).sum()
        return squared_norm.sqrt()

    def _flat_gradients(self, loss):
        parameters = self._optimizer_parameters()
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        pieces = []
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                pieces.append(
                    torch.zeros(
                        parameter.numel(),
                        device=loss.device,
                        dtype=loss.dtype,
                    )
                )
            else:
                pieces.append(gradient.detach().reshape(-1))
        if not pieces:
            return torch.zeros((), device=loss.device)
        return torch.cat(pieces)

    def _half_consistency_score(self, per_sample_loss, reliable_mask):
        """Interleaved half-gradient consistency for one loss component."""
        reliable_indices = reliable_mask.nonzero(as_tuple=False).view(-1)
        if reliable_indices.numel() < 4:
            return torch.zeros((), device=per_sample_loss.device)

        half_a = reliable_indices[::2]
        half_b = reliable_indices[1::2]
        loss_a = per_sample_loss.index_select(0, half_a).mean()
        loss_b = per_sample_loss.index_select(0, half_b).mean()
        gradient_a = self._flat_gradients(loss_a)
        gradient_b = self._flat_gradients(loss_b)
        norm_a = gradient_a.norm()
        norm_b = gradient_b.norm()
        cosine = torch.dot(gradient_a, gradient_b) / (
            norm_a * norm_b + 1e-8
        )
        norm_balance = (
            2.0 * torch.minimum(norm_a, norm_b)
            / (norm_a + norm_b + 1e-8)
        )
        return (
            cosine.clamp(0.0, 1.0) * norm_balance.clamp(0.0, 1.0)
        ).detach()

    @staticmethod
    def _append_reliable_values(destination, values, reliable):
        if reliable.any():
            destination.extend(values[reliable].detach().cpu().tolist())

    def _record_routing_diagnostics(
            self, q_proto, q_out, native_lambdas, relative_lambdas,
            selected_lambdas, reliable_mask, use_native):
        reliable = reliable_mask.bool()
        reliable_fraction = reliable_mask.float().mean()
        accepted_samples = int(reliable.sum().item())
        values = self.routing_diagnostic_values
        values['q_proto'].append(float(q_proto.item()))
        values['q_out'].append(float(q_out.item()))
        values['q_proto_over_q_out'].append(float(
            (q_proto / (q_out + 1e-8)).item()
        ))
        values['reliable_fraction'].append(float(reliable_fraction.item()))
        values['native_path_fraction'].append(float(use_native))
        values['accepted_samples'].append(float(accepted_samples))
        self._append_reliable_values(
            values['native_lambda'], native_lambdas, reliable
        )
        self._append_reliable_values(
            values['relative_lambda'], relative_lambdas, reliable
        )
        self._append_reliable_values(
            values['selected_lambda'], selected_lambdas, reliable
        )

    def _routed_lambdas(
            self, logits, similarities, pred_class, proto_identities,
            target_mask, reliable_mask, proto_loss_per_sample,
            output_loss_per_sample, routing_mode):
        """Build both candidates, calculate q, and select one batch path."""
        native_lambdas = self._controller_lambdas(similarities, target_mask)
        relative_lambdas = self._relative_margin_lambdas(
            logits, similarities, pred_class, proto_identities
        )
        relative_diagnostics = dict(self._last_controller_diagnostics)
        relative_output_evidence = relative_diagnostics['output_margin']

        q_proto = self._half_consistency_score(
            proto_loss_per_sample, reliable_mask
        )
        q_out = self._half_consistency_score(
            output_loss_per_sample, reliable_mask
        )
        reliable = reliable_mask.bool()
        reliable_fraction = reliable_mask.float().mean()
        mean_output_evidence = (
            relative_output_evidence[reliable].mean()
            if reliable.any()
            else torch.zeros((), device=logits.device)
        )
        coverage_native = bool(
            (mean_output_evidence >= 0.90).item()
            and (reliable_fraction <= 0.40).item()
        )

        if routing_mode == 'absolute_only_router':
            consistency_native = bool(
                (q_proto >= 2.0 * q_out).item()
                and (q_proto >= 0.25).item()
            )
            use_native = consistency_native
            coverage_native = False
        elif routing_mode == 'absolute_consistency_router':
            consistency_native = bool(
                (q_proto >= 2.0 * q_out).item()
                and (q_proto >= 0.25).item()
            )
            use_native = consistency_native or coverage_native
        elif routing_mode == 'forced_native':
            consistency_native = False
            use_native = True
        elif routing_mode in {'forced_relative', 'mean_act_rel'}:
            consistency_native = False
            use_native = False
        else:
            raise ValueError(f'Unsupported routing mode: {routing_mode}')

        if routing_mode == 'mean_act_rel':
            # Simplest label-free combination: sample-wise average of both estimates.
            selected_lambdas = 0.5 * (native_lambdas + relative_lambdas)
        else:
            selected_lambdas = (
                native_lambdas if use_native else relative_lambdas
            )
        selected_lambdas = selected_lambdas.detach().clamp(0.0, 1.0)

        self.coverage_router_total_batches += 1
        if use_native:
            self.coverage_router_native_batches += 1
        if consistency_native:
            self.coverage_router_gate_native += 1
        if coverage_native:
            self.coverage_router_coverage_native += 1
        self.coverage_router_q_proto_values.append(float(q_proto.item()))
        self.coverage_router_q_out_values.append(float(q_out.item()))
        self.coverage_router_reliable_fractions.append(
            float(reliable_fraction.item())
        )
        self._record_routing_diagnostics(
            q_proto, q_out, native_lambdas, relative_lambdas,
            selected_lambdas, reliable_mask, use_native
        )
        self._last_controller_diagnostics = {
            **relative_diagnostics,
            'q_proto': q_proto.detach(),
            'q_out': q_out.detach(),
            'use_native': float(use_native),
            'gate_native': float(consistency_native),
            'coverage_native': float(coverage_native),
            'reliable_fraction': reliable_fraction.detach(),
            'mean_output_evidence': mean_output_evidence.detach(),
        }
        return selected_lambdas

    def _controller_lambdas(self, similarities, target_mask):
        """Calculate detached lambda_i from top target-prototype activations."""
        with torch.no_grad():
            self._last_controller_diagnostics = None
            activations = similarities.detach()
            if activations.min() < 0:
                activations = (activations + 1.0) / 2.0
            activations = activations.clamp(0.0, 1.0)

            target_counts = target_mask.sum(dim=1)
            top_k = min(
                self.adaptive_top_k,
                int(target_counts.min().item()),
            )
            if top_k <= 0:
                raise RuntimeError('No target prototypes found for predicted class')
            target_activations = activations.masked_fill(
                ~target_mask.bool(),
                float('-inf'),
            )
            top_target = target_activations.topk(top_k, dim=1).values
            delta = (top_target - 0.5).abs().mean(dim=1)
            return (delta / self.adaptive_delta0).clamp(0.0, 1.0).detach()

    def _relative_margin_lambdas(
            self, logits, similarities, pred_class, proto_identities):
        """Compare prototype class margin with output probability margin."""
        with torch.no_grad():
            class_scores = []
            for class_index in range(logits.shape[1]):
                class_activations = similarities[
                    :, proto_identities == class_index
                ]
                top_k = min(
                    self.adaptive_top_k,
                    class_activations.shape[1],
                )
                class_scores.append(
                    class_activations.topk(top_k, dim=1).values.mean(dim=1)
                )
            class_scores = torch.stack(class_scores, dim=1)
            selected_score = class_scores.gather(
                1, pred_class.unsqueeze(1)
            ).squeeze(1)
            competing_score = class_scores.masked_fill(
                torch.nn.functional.one_hot(
                    pred_class, logits.shape[1]
                ).bool(),
                float('-inf'),
            ).max(dim=1).values
            proto_margin = (
                selected_score - competing_score
            ).clamp_min(0.0)
            relative_proto_margin = (
                proto_margin / selected_score.abs().clamp_min(1e-8)
            ).clamp(0.0, 1.0)

            output_probabilities = logits.softmax(dim=1)
            top_output_probabilities = output_probabilities.topk(
                2, dim=1
            ).values
            output_margin = (
                top_output_probabilities[:, 0]
                - top_output_probabilities[:, 1]
            ).clamp_min(0.0)
            relative_output_margin = (
                output_margin
                / top_output_probabilities[:, 0].clamp_min(1e-8)
            ).clamp(0.0, 1.0)
            lambdas = (
                relative_proto_margin
                / (
                    relative_proto_margin
                    + relative_output_margin
                    + 1e-8
                )
            ).clamp(0.0, 1.0)
            self._last_controller_diagnostics = {
                'proto_margin': relative_proto_margin.detach(),
                'output_margin': relative_output_margin.detach(),
                'agreement': (
                    class_scores.argmax(dim=1) == pred_class
                ).float().detach(),
            }
            return lambdas.detach()

    def _record_adaptive_diagnostics(
            self, lambdas, reliable_mask, proto_loss_per_sample,
            output_loss_per_sample, confidence_weights, proto_component,
            output_component, proto_gradient_norm, output_gradient_norm):
        reliable = reliable_mask.bool()
        reliable_count = int(reliable.sum().item())
        if reliable_count == 0:
            return
        reliable_weights = confidence_weights[reliable]
        self.lambda_values.extend(lambdas[reliable].detach().cpu().tolist())
        self.proto_loss_sum += float(
            (proto_loss_per_sample[reliable] * reliable_weights).sum().item()
        )
        self.output_loss_sum += float(
            (output_loss_per_sample[reliable] * reliable_weights).sum().item()
        )
        self.component_loss_sample_count += reliable_count
        self.proto_component_sum += float(proto_component.detach().item())
        self.output_component_sum += float(output_component.detach().item())
        self.component_batch_count += 1
        self.proto_gradient_norms.append(float(proto_gradient_norm.detach().item()))
        self.output_gradient_norms.append(float(output_gradient_norm.detach().item()))
        if self._last_controller_diagnostics is not None:
            self.proto_margin_values.extend(
                self._last_controller_diagnostics[
                    'proto_margin'
                ][reliable].cpu().tolist()
            )
            self.output_margin_values.extend(
                self._last_controller_diagnostics[
                    'output_margin'
                ][reliable].cpu().tolist()
            )
            self.proto_output_agreement_values.extend(
                self._last_controller_diagnostics[
                    'agreement'
                ][reliable].cpu().tolist()
            )

    def _native_normalized_similarities(self, min_distances):
        """Use the model's native distance-to-similarity mapping.

        The trained log-activation ProtoPNet historically scales its native
        scores by 9.0 for the binary-entropy loss and 0.8 reliability filter.
        Keeping that scaling preserves the established experiment protocol.
        """
        core = self.model.core if hasattr(self.model, 'core') else self.model
        if hasattr(core, 'distance_2_similarity'):
            raw_similarities = core.distance_2_similarity(min_distances)
        else:
            # Compatibility for minimal test doubles; production ProtoPNet
            # models always provide the native operation above.
            raw_similarities = torch.log(
                (min_distances + 1.0) / (min_distances + 1e-4)
            )
        activation = getattr(core, 'prototype_activation_function', None)
        if activation in {None, 'log'}:
            return raw_similarities / 9.0
        return raw_similarities

    def _compute_loss(self, logits, min_distances):
        """Compute samplewise lambda_i interpolation after per-sample losses."""
        device = logits.device
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        similarities = self._native_normalized_similarities(min_distances)
        detached_scores = similarities.detach()
        self.prototype_score_values.extend([
            float(detached_scores.min().item()),
            float(detached_scores.max().item()),
        ])
        proto_class_identity, last_layer = self._get_model_components()

        with torch.no_grad():
            pred_class = logits.argmax(dim=1)
            proto_identities = proto_class_identity.argmax(dim=1).to(device)

        reliable_mask = torch.ones(batch_size, device=device)
        margin_e0 = self.entropy_margin_scale * math.log(num_classes)
        entropy_mask = (
            softmax_entropy(logits) < margin_e0
        ).float()
        reliable_mask = reliable_mask * entropy_mask
        self.adaptation_stats['filtered_by_entropy'] += int(
            (1 - entropy_mask).sum().item()
        )
        max_sim_per_sample = similarities.max(dim=1)[0]
        geo_mask = (
            max_sim_per_sample > self.geo_filter_threshold
        ).float()
        reliable_mask = reliable_mask * geo_mask
        self.adaptation_stats['filtered_by_geo'] += int(
            (1 - geo_mask).sum().item()
        )
        self.geo_filter_stats['total_samples'] += batch_size
        self.geo_filter_stats['filtered_samples'] += int(
            (1 - geo_mask).sum().item()
        )

        if reliable_mask.sum() == 0:
            zero_loss = torch.tensor(0.0, device=device)
            return zero_loss, reliable_mask

        target_mask = (
            proto_identities.unsqueeze(0) == pred_class.unsqueeze(1)
        ).float()

        eps = 1e-6
        proto_probs = torch.clamp(
            similarities * target_mask,
            min=eps,
            max=1 - eps,
        )
        proto_entropy = -(
            proto_probs * torch.log(proto_probs)
            + (1 - proto_probs) * torch.log(1 - proto_probs)
        )
        proto_loss_per_sample = (
            (proto_entropy * target_mask).sum(dim=1)
            / (target_mask.sum(dim=1) + 1e-8)
        )
        output_loss_per_sample = softmax_entropy(logits)

        lambdas = self._routed_lambdas(
            logits,
            similarities,
            pred_class,
            proto_identities,
            target_mask,
            reliable_mask,
            proto_loss_per_sample,
            output_loss_per_sample,
            self.adaptive_controller,
        )

        # No confidence weighting on ProtoPNet.
        confidence_weights = torch.ones(batch_size, device=device)

        reliable_count = reliable_mask.sum()
        sample_weights = reliable_mask * confidence_weights

        # One scalar norm per raw component mean over exactly the accepted
        # samples and exactly the optimizer/adapted parameters. Lambda is not
        # included in either norm and no per-sample norms are calculated.
        proto_mean = (
            reliable_mask * proto_loss_per_sample
        ).sum() / (reliable_count + 1e-8)
        output_mean = (
            reliable_mask * output_loss_per_sample
        ).sum() / (reliable_count + 1e-8)
        proto_gradient_norm = self._gradient_norm(proto_mean)
        output_gradient_norm = self._gradient_norm(output_mean)

        if self.component_gradient_normalization:
            normalized_proto_loss = proto_loss_per_sample / (
                proto_gradient_norm.detach() + 1e-8
            )
            normalized_output_loss = output_loss_per_sample / (
                output_gradient_norm.detach() + 1e-8
            )
            sample_loss = (
                lambdas * normalized_proto_loss
                + (1.0 - lambdas) * normalized_output_loss
            )
            loss = (
                sample_weights * sample_loss
            ).sum() / (reliable_count + 1e-8)
            proto_component = (
                sample_weights * lambdas * normalized_proto_loss
            ).sum() / (reliable_count + 1e-8)
            output_component = (
                sample_weights * (1.0 - lambdas) * normalized_output_loss
            ).sum() / (reliable_count + 1e-8)
        else:
            proto_component = (
                sample_weights * lambdas * proto_loss_per_sample
            ).sum() / (reliable_count + 1e-8)
            output_component = (
                sample_weights * (1.0 - lambdas) * output_loss_per_sample
            ).sum() / (reliable_count + 1e-8)

            loss = proto_component + output_component

        with torch.no_grad():
            reliable = reliable_mask.bool()
            selected_weights = (
                torch.abs(last_layer.weight[pred_class]) * target_mask
            )
            mean_weight_per_sample = (
                selected_weights.sum(dim=1)
                / (target_mask.sum(dim=1) + 1e-8)
            )
            self.selected_prototype_weight_sum += float(
                mean_weight_per_sample[reliable].sum().item()
            )
            self.selected_prototype_weight_count += int(
                reliable.sum().item()
            )
        self._record_adaptive_diagnostics(
            lambdas,
            reliable_mask,
            proto_loss_per_sample,
            output_loss_per_sample,
            confidence_weights,
            proto_component,
            output_component,
            proto_gradient_norm,
            output_gradient_norm,
        )

        return loss, reliable_mask

    def get_stats(self):
        stats = super().get_stats()
        lambda_tensor = torch.tensor(self.lambda_values, dtype=torch.float64)
        proto_gradient_tensor = torch.tensor(
            self.proto_gradient_norms, dtype=torch.float64
        )
        output_gradient_tensor = torch.tensor(
            self.output_gradient_norms, dtype=torch.float64
        )
        proto_margin_tensor = torch.tensor(
            self.proto_margin_values, dtype=torch.float64
        )
        output_margin_tensor = torch.tensor(
            self.output_margin_values, dtype=torch.float64
        )
        agreement_tensor = torch.tensor(
            self.proto_output_agreement_values, dtype=torch.float64
        )
        if lambda_tensor.numel() > 0:
            lambda_quantiles = {
                str(quantile): float(torch.quantile(lambda_tensor, quantile).item())
                for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            }
            histogram = torch.histogram(
                lambda_tensor,
                bins=torch.linspace(0.0, 1.0, 11, dtype=torch.float64),
            ).hist.to(torch.int64)
            lambda_mean = float(lambda_tensor.mean().item())
            lambda_std = float(lambda_tensor.std(unbiased=False).item())
        else:
            lambda_quantiles = {}
            histogram = torch.zeros(10, dtype=torch.int64)
            lambda_mean = None
            lambda_std = None

        sample_count = self.component_loss_sample_count
        batch_count = self.component_batch_count
        q_proto_tensor = torch.tensor(
            self.coverage_router_q_proto_values, dtype=torch.float64
        )
        q_out_tensor = torch.tensor(
            self.coverage_router_q_out_values, dtype=torch.float64
        )
        reliable_fraction_tensor = torch.tensor(
            self.coverage_router_reliable_fractions, dtype=torch.float64
        )

        def distribution_summary(values):
            tensor = torch.tensor(values, dtype=torch.float64)
            if not tensor.numel():
                return {
                    'mean': None, 'p10': None, 'p50': None, 'p90': None,
                    'count': 0,
                }
            return {
                'mean': float(tensor.mean().item()),
                'p10': float(torch.quantile(tensor, 0.10).item()),
                'p50': float(torch.quantile(tensor, 0.50).item()),
                'p90': float(torch.quantile(tensor, 0.90).item()),
                'count': int(tensor.numel()),
            }

        routed_controllers = {
            'absolute_consistency_router', 'absolute_only_router',
            'forced_relative', 'forced_native',
        }
        prototype_score_summary = distribution_summary(
            self.prototype_score_values
        )
        stats.update({
            'adaptive_lambda': {
                'controller': self.adaptive_controller,
                'delta0': self.adaptive_delta0,
                'top_k': self.adaptive_top_k,
                'mean': lambda_mean,
                'std': lambda_std,
                'quantiles': lambda_quantiles,
                'histogram_bin_edges': [index / 10 for index in range(11)],
                'histogram_counts': histogram.tolist(),
                'sample_count': len(self.lambda_values),
            },
            'relative_margin_controller': {
                'mean_proto_margin': (
                    float(proto_margin_tensor.mean().item())
                    if proto_margin_tensor.numel() else None
                ),
                'mean_output_margin': (
                    float(output_margin_tensor.mean().item())
                    if output_margin_tensor.numel() else None
                ),
                'proto_output_agreement_rate': (
                    float(agreement_tensor.mean().item())
                    if agreement_tensor.numel() else None
                ),
                'definition': (
                    'lambda_i=relative_proto_class_margin/'
                    '(relative_proto_class_margin+relative_output_margin); '
                    'each margin is normalized by its winning score'
                ),
            },
            'coverage_router': {
                'enabled': self.adaptive_controller in routed_controllers,
                'controller': self.adaptive_controller,
                'native_batch_fraction': (
                    self.coverage_router_native_batches
                    / self.coverage_router_total_batches
                    if self.coverage_router_total_batches else None
                ),
                'gate_native_batch_fraction': (
                    self.coverage_router_gate_native
                    / self.coverage_router_total_batches
                    if self.coverage_router_total_batches else None
                ),
                'coverage_native_batch_fraction': (
                    self.coverage_router_coverage_native
                    / self.coverage_router_total_batches
                    if self.coverage_router_total_batches else None
                ),
                'mean_q_proto': (
                    float(q_proto_tensor.mean().item())
                    if q_proto_tensor.numel() else None
                ),
                'mean_q_out': (
                    float(q_out_tensor.mean().item())
                    if q_out_tensor.numel() else None
                ),
                'mean_reliable_fraction': (
                    float(reliable_fraction_tensor.mean().item())
                    if reliable_fraction_tensor.numel() else None
                ),
                'total_batches': self.coverage_router_total_batches,
                'definition': (
                    'absolute_only_router uses q_proto>=2*q_out and '
                    'q_proto>=0.25; absolute_consistency_router additionally '
                    'uses mean r_o>=0.90 and reliable coverage<=0.40'
                ),
            },
            'routing_diagnostics': {
                key: distribution_summary(values)
                for key, values in self.routing_diagnostic_values.items()
            },
            'prototype_scores': {
                **prototype_score_summary,
                'min': (
                    min(self.prototype_score_values)
                    if self.prototype_score_values else None
                ),
                'max': (
                    max(self.prototype_score_values)
                    if self.prototype_score_values else None
                ),
                'larger_means_more_similar': True,
                'distance_to_similarity': 'model native operation',
                'normalization': (
                    'native log similarity divided by 9.0, matching the '
                    'prototype entropy and reliability filter'
                ),
                'reliability_threshold': self.geo_filter_threshold,
            },
            'accepted_sample_percentage': (
                100.0 * stats['adapted_samples'] / stats['total_samples']
                if stats['total_samples'] else 0.0
            ),
            'mean_proto_loss': (
                self.proto_loss_sum / sample_count if sample_count else None
            ),
            'mean_output_loss': (
                self.output_loss_sum / sample_count if sample_count else None
            ),
            'mean_lambda_weighted_proto_component': (
                self.proto_component_sum / batch_count if batch_count else None
            ),
            'mean_lambda_weighted_output_component': (
                self.output_component_sum / batch_count if batch_count else None
            ),
            'mean_proto_gradient_norm': (
                float(proto_gradient_tensor.mean().item())
                if proto_gradient_tensor.numel() else None
            ),
            'std_proto_gradient_norm': (
                float(proto_gradient_tensor.std(unbiased=False).item())
                if proto_gradient_tensor.numel() else None
            ),
            'mean_output_gradient_norm': (
                float(output_gradient_tensor.mean().item())
                if output_gradient_tensor.numel() else None
            ),
            'std_output_gradient_norm': (
                float(output_gradient_tensor.std(unbiased=False).item())
                if output_gradient_tensor.numel() else None
            ),
            'component_gradient_batch_count': batch_count,
            'component_gradient_normalization': (
                self.component_gradient_normalization
            ),
            'gradient_normalization_definition': (
                'one detached L2 norm of mean reliable raw prototype loss '
                'and one of mean reliable raw output loss over optimizer '
                'parameters; each scalar divides every sample of its component'
            ),
        })
        return stats


# ============================================================================
# Setup Functions
# ============================================================================

def collect_params_enhanced(model, adaptation_mode='batchnorm_addon'):
    """Collect parameters for enhanced ProtoEntropy."""
    params = []
    names = []

    if hasattr(model, 'core'):
        core = model.core
        prefix = 'core.'
    else:
        core = model
        prefix = ''

    # BatchNorm params
    if 'batchnorm' in adaptation_mode:
        for nm, m in core.features.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{prefix}features.{nm}.{np_name}")

        for nm, m in core.add_on_layers.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{prefix}add_on_layers.{nm}.{np_name}")

    # Add-on layer params
    if 'addon' in adaptation_mode:
        for nm, p in core.add_on_layers.named_parameters():
            param_name = f"{prefix}add_on_layers.{nm}"
            if p.requires_grad and param_name not in names:
                params.append(p)
                names.append(param_name)

    return params, names


def configure_model_enhanced(model, adaptation_mode='batchnorm_addon'):
    """Configure model for enhanced ProtoEntropy."""
    if hasattr(model, 'core'):
        core = model.core
    else:
        core = model

    model.train()
    model.requires_grad_(False)

    if 'batchnorm' in adaptation_mode:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)

    if 'addon' in adaptation_mode:
        for p in core.add_on_layers.parameters():
            p.requires_grad = True

    return model


def setup_proto_entropy_enhanced(model,
                                  lr=0.001,
                                  alpha_proto=1.0,
                                  alpha_softmax=0.0,
                                  entropy_margin_scale=0.4,
                                  geo_filter_threshold=0.3,
                                  adaptation_mode='batchnorm_addon',
                                  steps=1,
                                  samplewise_adaptive_lambda=False,
                                  adaptive_delta0=0.25,
                                  adaptive_top_k=3,
                                  component_gradient_normalization=False,
                                  adaptive_controller='absolute_consistency_router'):
    """
    Set up ProtoTTA on ProtoPNet: a fixed prototype weight (alpha_proto) or,
    with samplewise_adaptive_lambda=True, the label-free per-sample controller.
    Both use the entropy and geometric reliability filters and Adam.
    """
    model = configure_model_enhanced(model, adaptation_mode=adaptation_mode)
    params, param_names = collect_params_enhanced(model, adaptation_mode=adaptation_mode)

    if not params:
        print(f"Warning: No parameters found for mode {adaptation_mode}")
        return model

    # Create optimizer
    optimizer = torch.optim.Adam(params, lr=lr)

    if samplewise_adaptive_lambda:
        proto_model = SamplewiseAdaptiveProtoEntropy(
            model,
            optimizer,
            steps=steps,
            alpha_proto=alpha_proto,
            alpha_softmax=alpha_softmax,
            entropy_margin_scale=entropy_margin_scale,
            geo_filter_threshold=geo_filter_threshold,
            adaptive_delta0=adaptive_delta0,
            adaptive_top_k=adaptive_top_k,
            component_gradient_normalization=component_gradient_normalization,
            adaptive_controller=adaptive_controller,
        )
    else:
        proto_model = ProtoEntropyEnhanced(
            model,
            optimizer,
            steps=steps,
            alpha_proto=alpha_proto,
            alpha_softmax=alpha_softmax,
            entropy_margin_scale=entropy_margin_scale,
            geo_filter_threshold=geo_filter_threshold,
        )

    mode_str = [f"ent@{entropy_margin_scale}", f"geo@{geo_filter_threshold}"]
    if alpha_softmax > 0:
        mode_str.append(f"hybrid({alpha_proto:.1f}:{alpha_softmax:.1f})")
    if samplewise_adaptive_lambda:
        mode_str.append(
            f"sample-lambda({adaptive_controller},k={adaptive_top_k},"
            f"delta0={adaptive_delta0})"
        )
    if component_gradient_normalization:
        mode_str.append("component-gradnorm")

    print(f"ProtoEntropy++ ({'+'.join(mode_str)}, {adaptation_mode}, {len(params)} params)")

    return proto_model
