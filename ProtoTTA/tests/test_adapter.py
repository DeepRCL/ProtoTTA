import math
import unittest
from unittest.mock import patch

import torch
from torch import nn

from ProtoTTA import (
    ProtoTTAAdapter,
    ProtoTTAConfig,
    PrototypeBatch,
    configure_model,
    default_output_extractor,
)


class TinyPrototypeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(3)
        self.prototypes = nn.Parameter(torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]))
        self.register_buffer("identity", torch.eye(2))

    def forward(self, inputs):
        features = torch.nn.functional.normalize(self.norm(inputs), dim=1)
        prototypes = torch.nn.functional.normalize(self.prototypes, dim=1)
        scores = features @ prototypes.t()
        return {
            "logits": 3.0 * scores,
            "prototype_scores": scores,
            "filter_scores": scores,
        }


INPUTS = torch.tensor([
    [0.2, 1.0, -0.5],
    [1.2, -0.3, 0.1],
    [-0.4, 0.7, 0.2],
    [0.8, -0.1, -0.6],
    [0.3, 0.9, -1.1],
    [1.5, 0.2, -0.4],
])


def build_adapter(*, config=None, lr=0.05):
    model = TinyPrototypeModel()
    parameters, names = configure_model(model)
    optimizer = torch.optim.SGD(parameters, lr=lr)
    adapter = ProtoTTAAdapter(
        model,
        optimizer,
        prototype_class_identity=model.identity,
        config=config or ProtoTTAConfig(use_geometric_filter=False),
    )
    return model, adapter, names


def output_entropy(logits):
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


class ConfigurationTest(unittest.TestCase):
    def test_defaults_follow_appendix_l(self):
        config = ProtoTTAConfig()
        self.assertEqual(config.score_mapping, "cosine")
        self.assertEqual(config.aggregation, "topk_mean")
        self.assertEqual(config.aggregation_topk, 2)
        self.assertTrue(config.use_geometric_filter)
        self.assertEqual(config.geometric_threshold, 0.92)
        self.assertIsNone(config.entropy_filter_factor)
        self.assertTrue(config.use_confidence_weighting)
        self.assertTrue(config.use_importance_weighting)
        self.assertFalse(config.adapt_all_prototypes)
        self.assertEqual(config.interpolation_mode, "adaptive")
        self.assertTrue(config.gradient_normalize)
        self.assertEqual(config.controller_topk, 3)
        self.assertEqual(config.controller_delta0, 0.25)
        self.assertEqual(config.controller_consistency_ratio, 2.0)
        self.assertEqual(config.controller_min_consistency, 0.25)
        self.assertEqual(config.controller_min_reliable, 4)
        self.assertEqual(config.controller_output_floor, 1e-8)
        self.assertTrue(config.use_coverage_guard)
        self.assertEqual(config.coverage_output_margin, 0.90)
        self.assertEqual(config.coverage_max_fraction, 0.40)
        self.assertEqual(config.gradient_epsilon, 1e-12)
        self.assertEqual(config.entropy_epsilon, 1e-6)
        self.assertEqual(config.importance_epsilon, 1e-6)
        self.assertEqual(config.eps, 1e-8)

    def test_fixed_lambda_accepts_both_endpoints(self):
        for value in (0.0, 1.0):
            self.assertEqual(
                ProtoTTAConfig(interpolation_mode="fixed", fixed_lambda=value).fixed_lambda,
                value,
            )
        with self.assertRaises(ValueError):
            ProtoTTAConfig(fixed_lambda=1.5)

    def test_configure_model_selects_only_normalization(self):
        model, _, names = build_adapter()
        self.assertEqual(names, ["norm.weight", "norm.bias"])
        self.assertFalse(model.prototypes.requires_grad)

    def test_extra_patterns_select_attention_biases(self):
        model = nn.Module()
        model.norm = nn.LayerNorm(4)
        model.attn = nn.Module()
        model.attn.qkv = nn.Linear(4, 12)
        model.attn.proj = nn.Linear(4, 4)
        model.head = nn.Linear(4, 2)
        _, names = configure_model(model, extra_parameter_patterns=("*attn*bias",))
        self.assertEqual(
            names,
            ["norm.weight", "norm.bias", "attn.qkv.bias", "attn.proj.bias"],
        )
        self.assertFalse(model.attn.qkv.weight.requires_grad)
        self.assertFalse(model.head.bias.requires_grad)

    def test_batchnorm_uses_current_batch_statistics_in_any_mode(self):
        module = nn.Sequential(nn.BatchNorm1d(3))
        with torch.no_grad():
            module[0].running_mean.fill_(5.0)
            module[0].running_var.fill_(9.0)
        _, names = configure_model(
            module,
            adapt_layer_norm=False,
            use_batch_statistics=True,
        )
        self.assertEqual(names, ["0.weight", "0.bias"])
        self.assertFalse(module[0].track_running_stats)
        self.assertIsNone(module[0].running_mean)
        self.assertIsNone(module[0].running_var)
        inputs = torch.randn(8, 3)
        expected = (inputs - inputs.mean(dim=0)) / torch.sqrt(
            inputs.var(dim=0, unbiased=False) + module[0].eps
        )
        self.assertTrue(torch.allclose(module(inputs), expected, atol=1e-5))
        module.eval()
        self.assertTrue(torch.allclose(module(inputs), expected, atol=1e-5))

    def test_batch_statistics_state_survives_reset(self):
        model = nn.Sequential(nn.BatchNorm1d(2))
        parameters, _ = configure_model(
            model, adapt_layer_norm=False, use_batch_statistics=True
        )

        def extract(output):
            return PrototypeBatch(
                logits=output, prototype_scores=output.tanh(),
                reliability_mask=torch.ones(output.shape[0], dtype=torch.bool),
            )

        adapter = ProtoTTAAdapter(
            model,
            torch.optim.SGD(parameters, lr=0.1),
            prototype_class_identity=torch.eye(2),
            output_extractor=extract,
            config=ProtoTTAConfig(interpolation_mode="fixed"),
        )
        before = model[0].weight.detach().clone()
        adapter(torch.randn(6, 2))
        adapter.reset()
        self.assertTrue(torch.equal(model[0].weight, before))
        self.assertIsNone(model[0].running_mean)
        self.assertIsNone(model[0].running_var)
        self.assertIsNone(model[0].num_batches_tracked)
        self.assertFalse(model[0].track_running_stats)
        model.eval()
        inputs = torch.randn(8, 2)
        expected = (inputs - inputs.mean(dim=0)) / torch.sqrt(
            inputs.var(dim=0, unbiased=False) + model[0].eps
        )
        self.assertTrue(torch.allclose(model(inputs), expected, atol=1e-5))


class ActivationAndFilterTest(unittest.TestCase):
    def test_subprototype_aggregation(self):
        _, adapter, _ = build_adapter()
        scores = torch.tensor([[[0.1, 0.7, 0.3], [0.2, 0.5, 0.4]]])
        aggregated = adapter._aggregate(scores)
        self.assertTrue(torch.allclose(aggregated, torch.tensor([[0.5, 0.45]])))
        _, max_adapter, _ = build_adapter(
            config=ProtoTTAConfig(aggregation="max", use_geometric_filter=False)
        )
        self.assertTrue(torch.equal(
            max_adapter._aggregate(scores), torch.tensor([[0.7, 0.5]])
        ))

    def test_cosine_mapping_and_clamps(self):
        _, adapter, _ = build_adapter()
        scores = torch.tensor([[-1.0, 0.0, 1.0]])
        values = adapter._activation_values(scores)
        self.assertTrue(torch.equal(values, torch.tensor([[0.0, 0.5, 1.0]])))
        loss_values = adapter._map_scores(scores)
        self.assertAlmostEqual(loss_values[0, 0].item(), 1e-6, places=12)
        self.assertAlmostEqual(loss_values[0, 2].item(), 1.0 - 1e-6, places=6)
        batch = PrototypeBatch(logits=torch.zeros(1, 2), prototype_scores=scores)
        controller = adapter._controller_activations(batch, values)
        self.assertAlmostEqual(controller[0, 0].item(), 1e-8, places=14)
        self.assertEqual(controller[0, 2].item(), 1.0)

    def test_distance_mapping_matches_log_inverse_kernel(self):
        _, adapter, _ = build_adapter(
            config=ProtoTTAConfig(score_mapping="distance", use_geometric_filter=False)
        )
        distances = torch.tensor([[0.0, 1.0, 10.0]])
        native = adapter._native_scores(distances)
        expected = torch.log((distances + 1.0) / (distances + 1e-4)) / 9.0
        self.assertTrue(torch.allclose(native, expected))
        self.assertGreater(native[0, 0].item(), 1.0)
        mapped = adapter._map_scores(distances)
        self.assertAlmostEqual(mapped[0, 0].item(), 1.0 - 1e-6, places=6)
        self.assertAlmostEqual(mapped[0, 1].item(), 0.0770, places=3)
        batch = PrototypeBatch(logits=torch.zeros(1, 2), prototype_scores=distances)
        controller = adapter._controller_activations(
            batch, adapter._activation_values(distances)
        )
        self.assertEqual(controller[0, 0].item(), 1.0)

    def test_geometric_filter_uses_raw_aggregated_score(self):
        _, adapter, _ = build_adapter(config=ProtoTTAConfig())
        scores = torch.tensor([
            [[0.95, 0.94, 0.10, 0.00], [0.20, 0.10, 0.00, 0.00]],
            [[0.99, 0.80, 0.10, 0.00], [0.20, 0.10, 0.00, 0.00]],
        ])
        batch = PrototypeBatch(logits=torch.zeros(2, 2), prototype_scores=scores)
        reliable = adapter._reliable_mask(batch, adapter._aggregate(scores))
        self.assertEqual(reliable.tolist(), [True, False])

    def test_entropy_filter_factor(self):
        config = ProtoTTAConfig(
            use_geometric_filter=False, entropy_filter_factor=0.4
        )
        _, adapter, _ = build_adapter(config=config)
        logits = torch.tensor([[6.0, 0.0], [0.1, 0.0]])
        batch = PrototypeBatch(logits=logits, prototype_scores=torch.zeros(2, 2))
        reliable = adapter._reliable_mask(batch, torch.zeros(2, 2))
        margin = 0.4 * math.log(2)
        self.assertEqual(reliable.tolist(), (output_entropy(logits) < margin).tolist())
        self.assertEqual(reliable.tolist(), [True, False])

    def test_filter_can_skip_an_entire_batch(self):
        model, adapter, _ = build_adapter(
            config=ProtoTTAConfig(geometric_threshold=1.0)
        )
        before = model.norm.weight.detach().clone()
        adapter(torch.randn(3, 3))
        self.assertTrue(torch.equal(before, model.norm.weight.detach()))
        self.assertEqual(adapter.stats["adapted_samples"], 0)
        self.assertEqual(adapter.stats["optimizer_steps"], 0)


class ObjectiveTest(unittest.TestCase):
    def test_forward_updates_and_reset_restores_parameters(self):
        model, adapter, _ = build_adapter()
        before = model.norm.weight.detach().clone()
        output = adapter(INPUTS)
        self.assertEqual(tuple(output["logits"].shape), (6, 2))
        self.assertFalse(torch.equal(before, model.norm.weight.detach()))
        self.assertEqual(adapter.stats["optimizer_steps"], 1)
        adapter.reset()
        self.assertTrue(torch.equal(before, model.norm.weight.detach()))

    def test_fixed_mode_uses_paper_lambda_convention(self):
        config = ProtoTTAConfig(
            interpolation_mode="fixed",
            fixed_lambda=0.7,
            use_geometric_filter=False,
        )
        _, adapter, _ = build_adapter(config=config)
        adapter(INPUTS)
        self.assertAlmostEqual(adapter.stats["last_lambda_mean"], 0.7, places=6)

    def test_fixed_mode_interpolates_prototype_then_output_loss(self):
        config = ProtoTTAConfig(
            interpolation_mode="fixed",
            fixed_lambda=0.7,
            gradient_normalize=False,
            use_confidence_weighting=False,
            use_geometric_filter=False,
        )
        model, adapter, _ = build_adapter(config=config)
        batch = adapter.output_extractor(model(INPUTS))
        probabilities = adapter._map_scores(adapter._aggregate(batch.prototype_scores))
        prototype_loss = adapter._prototype_loss_per_sample(
            batch, probabilities, adapter._routing_mask(batch, probabilities)
        )
        expected = (
            0.7 * prototype_loss + 0.3 * output_entropy(batch.logits)
        ).sum() / (len(INPUTS) + config.eps)

        adapter(INPUTS)
        self.assertAlmostEqual(adapter.stats["last_loss"], expected.item(), places=6)

    def test_both_terms_share_the_confidence_weight(self):
        config = ProtoTTAConfig(
            interpolation_mode="fixed",
            fixed_lambda=0.0,
            gradient_normalize=False,
            use_geometric_filter=False,
        )
        model, adapter, _ = build_adapter(config=config)
        logits = model(INPUTS)["logits"]
        confidence = logits.softmax(dim=1).max(dim=1).values
        expected = (confidence * output_entropy(logits)).sum() / (len(INPUTS) + config.eps)
        adapter(INPUTS)
        self.assertAlmostEqual(adapter.stats["last_loss"], expected.item(), places=6)

    def test_shared_prototypes_use_soft_target_weighted_bce(self):
        model = TinyPrototypeModel()
        parameters, _ = configure_model(model)
        weights = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
        adapter = ProtoTTAAdapter(
            model,
            torch.optim.SGD(parameters, lr=0.01),
            classifier_weights=weights,
            config=ProtoTTAConfig(interpolation_mode="fixed"),
        )
        probabilities = torch.tensor([[0.8, 0.3]])
        batch = PrototypeBatch(
            logits=torch.tensor([[3.0, 1.0]]),
            prototype_scores=probabilities,
        )
        routing = adapter._routing_mask(batch, probabilities)
        loss = adapter._prototype_loss_per_sample(batch, probabilities, routing)
        targets = torch.sigmoid(torch.tensor([[4.0, -2.0]]))
        importance = torch.tensor([[2.0, 1.0]]) / (2.0 + 1e-6)
        expected = (
            importance
            * -(
                targets * probabilities.log()
                + (1.0 - targets) * (1.0 - probabilities).log()
            )
        ).mean(dim=1)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.equal(routing, torch.tensor([[1.0, 0.0]])))

    def test_soft_target_routing_requires_positive_support(self):
        model = TinyPrototypeModel()
        parameters, _ = configure_model(model)
        adapter = ProtoTTAAdapter(
            model,
            torch.optim.SGD(parameters, lr=0.01),
            config=ProtoTTAConfig(interpolation_mode="fixed"),
        )
        weights = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
        probabilities = torch.full((2, 3), 0.5)
        batch = PrototypeBatch(
            logits=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            prototype_scores=probabilities,
            prototype_targets=torch.sigmoid(2.0 * weights),
        )
        routing = adapter._routing_mask(batch, probabilities)
        # Zero-weight targets (0.5) are excluded; an empty row falls back to all.
        self.assertTrue(torch.equal(
            routing, torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        ))

    def test_per_batch_target_mask_needs_no_static_routing(self):
        model = TinyPrototypeModel()
        parameters, _ = configure_model(model)
        optimizer = torch.optim.SGD(parameters, lr=0.01)

        def extract_with_mask(output):
            return default_output_extractor({
                **output,
                "target_mask": torch.ones_like(output["prototype_scores"]),
            })

        adapter = ProtoTTAAdapter(
            model,
            optimizer,
            output_extractor=extract_with_mask,
            config=ProtoTTAConfig(
                interpolation_mode="fixed",
                use_geometric_filter=False,
            ),
        )
        adapter(torch.randn(2, 3))
        self.assertEqual(adapter.stats["optimizer_steps"], 1)

    def test_adaptive_update_matches_normalized_objective(self):
        """One SGD step equals -lr times the gradient of Eq. 61."""

        config = ProtoTTAConfig(use_geometric_filter=False)
        lr = 0.05
        model, adapter, _ = build_adapter(config=config, lr=lr)
        reference = TinyPrototypeModel()
        reference.load_state_dict(model.state_dict())
        ref_parameters, _ = configure_model(reference)

        output = reference(INPUTS)
        batch = default_output_extractor(output)
        values = adapter._activation_values(batch.prototype_scores)
        probabilities = values.clamp(1e-6, 1.0 - 1e-6)
        routing = adapter._routing_mask(batch, probabilities)
        proto = adapter._prototype_loss_per_sample(batch, probabilities, routing)
        out = output_entropy(batch.logits)
        confidence = batch.logits.softmax(dim=1).max(dim=1).values.detach()
        denominator = len(INPUTS) + config.eps

        def norm(loss):
            grads = torch.autograd.grad(loss, ref_parameters, retain_graph=True)
            return torch.sqrt(sum(g.pow(2).sum() for g in grads)).detach()

        g_proto = norm((confidence * proto).sum() / denominator)
        g_out = norm((confidence * out).sum() / denominator)
        reliable = torch.ones(len(INPUTS), dtype=torch.bool)
        lam, _ = adapter._adaptive_lambda(
            batch, values.detach().clamp(1e-8, 1.0), routing, proto, out,
            confidence, reliable, ref_parameters,
        )
        objective = (
            confidence * (
                lam * proto / (g_proto + config.eps)
                + (1.0 - lam) * out / (g_out + config.eps)
            )
        ).sum() / denominator
        expected_grads = torch.autograd.grad(objective, ref_parameters)

        before = [p.detach().clone() for p in adapter._adaptation_parameters()]
        adapter(INPUTS)
        after = adapter._adaptation_parameters()
        for start, end, grad in zip(before, after, expected_grads):
            self.assertTrue(torch.allclose(end.detach(), start - lr * grad, atol=1e-6))
        self.assertAlmostEqual(adapter.stats["last_loss"], objective.item(), places=6)


class ControllerTest(unittest.TestCase):
    def test_activation_margin_coefficient_matches_paper(self):
        _, adapter, _ = build_adapter()
        activations = torch.tensor([[0.9, 0.8], [0.55, 0.45]])
        eligible = torch.ones_like(activations)
        batch = PrototypeBatch(
            logits=torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
            prototype_scores=activations,
        )
        coefficient = adapter._activation_lambda(batch, activations, eligible)
        expected = torch.tensor([1.0, 0.2])
        self.assertTrue(torch.allclose(coefficient, expected, atol=1e-6))

    def test_relative_evidence_coefficient_matches_definition(self):
        _, adapter, _ = build_adapter()
        activations = torch.tensor([[0.9, 0.4]])
        batch = PrototypeBatch(
            logits=torch.tensor([[2.0, 0.0]]),
            prototype_scores=activations,
        )
        coefficient, proto_margin, output_margin = adapter._relative_lambda(
            batch, activations
        )
        expected = proto_margin / (proto_margin + output_margin + adapter.config.eps)
        self.assertTrue(torch.allclose(coefficient, expected))
        self.assertAlmostEqual(proto_margin.item(), (0.9 - 0.4) / 0.9, places=6)
        top = torch.tensor([2.0, 0.0]).softmax(dim=0)
        self.assertAlmostEqual(
            output_margin.item(), ((top[0] - top[1]) / top[0]).item(), places=6
        )

    def test_coefficients_reach_both_endpoints(self):
        _, adapter, _ = build_adapter()
        activations = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        batch = PrototypeBatch(
            logits=torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
            prototype_scores=activations,
        )
        eligible = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.assertTrue(torch.equal(
            adapter._activation_lambda(batch, activations, eligible),
            torch.ones(2),
        ))
        flat = torch.full((2, 2), 0.6)
        relative, _, _ = adapter._relative_lambda(batch, flat)
        self.assertTrue(torch.equal(relative, torch.zeros(2)))

    def test_controller_uses_pseudo_label_prototypes_when_adapting_all(self):
        stats = []
        for adapt_all in (False, True):
            config = ProtoTTAConfig(
                use_geometric_filter=False, adapt_all_prototypes=adapt_all
            )
            _, adapter, _ = build_adapter(config=config)
            adapter(INPUTS)
            stats.append(adapter.stats["last_activation_lambda_mean"])
        self.assertAlmostEqual(stats[0], stats[1], places=7)

    def test_adaptive_router_selects_activation_margin_when_consistent(self):
        config = ProtoTTAConfig(
            use_coverage_guard=False,
            use_geometric_filter=False,
        )
        model, adapter, _ = build_adapter(config=config)
        activations = torch.tensor([
            [0.9, 0.2], [0.8, 0.3], [0.7, 0.4], [0.95, 0.1]
        ])
        batch = PrototypeBatch(
            logits=torch.tensor([[2.0, 0.0]] * 4),
            prototype_scores=activations,
        )
        eligible = torch.tensor([[1.0, 0.0]] * 4)
        sample_loss = torch.ones(4, requires_grad=True)
        reliable = torch.ones(4, dtype=torch.bool)
        confidence = torch.ones(4)
        with patch.object(
            adapter,
            "_gradient_consistency",
            side_effect=[torch.tensor(0.5), torch.tensor(0.1)],
        ):
            coefficient, diagnostics = adapter._adaptive_lambda(
                batch, activations, eligible, sample_loss, sample_loss,
                confidence, reliable, [model.norm.weight],
            )
        expected = adapter._activation_lambda(batch, activations, eligible)
        self.assertTrue(torch.allclose(coefficient, expected))
        self.assertEqual(diagnostics["router_gate"].item(), 1.0)

    def test_router_requires_absolute_consistency(self):
        config = ProtoTTAConfig(
            use_coverage_guard=False,
            use_geometric_filter=False,
        )
        model, adapter, _ = build_adapter(config=config)
        activations = torch.tensor([[0.9, 0.2]] * 4)
        batch = PrototypeBatch(
            logits=torch.tensor([[2.0, 0.0]] * 4),
            prototype_scores=activations,
        )
        eligible = torch.tensor([[1.0, 0.0]] * 4)
        sample_loss = torch.ones(4, requires_grad=True)
        with patch.object(
            adapter,
            "_gradient_consistency",
            side_effect=[torch.tensor(0.2), torch.tensor(0.0)],
        ):
            coefficient, diagnostics = adapter._adaptive_lambda(
                batch, activations, eligible, sample_loss, sample_loss,
                torch.ones(4), torch.ones(4, dtype=torch.bool), [model.norm.weight],
            )
        relative, _, _ = adapter._relative_lambda(batch, activations)
        self.assertTrue(torch.allclose(coefficient, relative))
        self.assertEqual(diagnostics["router_gate"].item(), 0.0)

    def test_gradient_consistency_matches_identical_half_gradients(self):
        model, adapter, _ = build_adapter()
        parameter = model.norm.weight
        losses = torch.stack([
            parameter[0], parameter[0], 2.0 * parameter[0], 2.0 * parameter[0]
        ])
        consistency = adapter._gradient_consistency(
            losses,
            torch.ones(4),
            torch.ones(4, dtype=torch.bool),
            [parameter],
        )
        self.assertAlmostEqual(consistency.item(), 1.0, places=6)

    def test_gradient_consistency_needs_four_reliable_samples(self):
        model, adapter, _ = build_adapter()
        parameter = model.norm.weight
        losses = torch.stack([parameter[0]] * 3)
        consistency = adapter._gradient_consistency(
            losses, torch.ones(3), torch.ones(3, dtype=torch.bool), [parameter]
        )
        self.assertEqual(consistency.item(), 0.0)

    def test_coverage_guard_can_select_activation_margin(self):
        config = ProtoTTAConfig(use_geometric_filter=False)
        model, adapter, _ = build_adapter(config=config)
        activations = torch.tensor([
            [0.9, 0.2], [0.8, 0.3], [0.7, 0.4], [0.95, 0.1]
        ])
        batch = PrototypeBatch(
            logits=torch.tensor([[10.0, 0.0]] * 4),
            prototype_scores=activations,
        )
        eligible = torch.tensor([[1.0, 0.0]] * 4)
        reliable = torch.tensor([True, False, False, False])
        sample_loss = torch.ones(4, requires_grad=True)
        with patch.object(
            adapter,
            "_gradient_consistency",
            side_effect=[torch.tensor(0.0), torch.tensor(0.0)],
        ):
            coefficient, diagnostics = adapter._adaptive_lambda(
                batch, activations, eligible, sample_loss, sample_loss,
                torch.ones(4), reliable, [model.norm.weight],
            )
        expected = adapter._activation_lambda(batch, activations, eligible)
        self.assertTrue(torch.allclose(coefficient, expected))
        self.assertEqual(diagnostics["router_gate"].item(), 1.0)
        self.assertAlmostEqual(diagnostics["coverage"].item(), 0.25)

    def test_adaptive_forward_records_controller_diagnostics(self):
        _, adapter, _ = build_adapter()
        adapter(INPUTS)
        self.assertIsNotNone(adapter.stats["last_activation_lambda_mean"])
        self.assertIsNotNone(adapter.stats["last_relative_lambda_mean"])
        self.assertIn(adapter.stats["last_router_gate"], {0, 1})
        self.assertGreaterEqual(adapter.stats["last_lambda_mean"], 0.0)
        self.assertLessEqual(adapter.stats["last_lambda_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
