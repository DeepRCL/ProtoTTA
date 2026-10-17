import unittest
from unittest.mock import patch

import torch
from torch import nn

from ProtoTTA import (
    ProtoTTAAdapter,
    ProtoTTAConfig,
    PrototypeBatch,
    configure_model,
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
            "logits": scores,
            "prototype_scores": scores,
            "filter_scores": scores,
        }


def build_adapter(*, threshold=0.0, config=None):
    model = TinyPrototypeModel()
    parameters, names = configure_model(model)
    optimizer = torch.optim.SGD(parameters, lr=0.05)
    adapter = ProtoTTAAdapter(
        model,
        optimizer,
        prototype_class_identity=model.identity,
        config=config or ProtoTTAConfig(geometric_threshold=threshold),
    )
    return model, adapter, names


class ProtoTTAAdapterTest(unittest.TestCase):
    def test_configure_model_selects_only_normalization(self):
        model, _, names = build_adapter()
        self.assertEqual(names, ["norm.weight", "norm.bias"])
        self.assertFalse(model.prototypes.requires_grad)

    def test_batchnorm_can_use_batch_stats_without_running_updates(self):
        module = nn.Sequential(nn.BatchNorm1d(3))
        running_before = module[0].running_mean.clone()
        _, names = configure_model(
            module,
            adapt_layer_norm=False,
            use_batch_statistics=True,
        )
        module(torch.randn(4, 3))
        self.assertEqual(names, ["0.weight", "0.bias"])
        self.assertTrue(module[0].training)
        self.assertFalse(module[0].track_running_stats)
        self.assertTrue(torch.equal(module[0].running_mean, running_before))

    def test_forward_updates_and_reset_restores_parameters(self):
        torch.manual_seed(2)
        model, adapter, _ = build_adapter()
        inputs = torch.tensor([[0.2, 1.0, -0.5], [1.2, -0.3, 0.1]])
        before = model.norm.weight.detach().clone()
        output = adapter(inputs)
        self.assertEqual(tuple(output["logits"].shape), (2, 2))
        self.assertFalse(torch.equal(before, model.norm.weight.detach()))
        self.assertEqual(adapter.stats["optimizer_steps"], 1)
        adapter.reset()
        self.assertTrue(torch.equal(before, model.norm.weight.detach()))

    def test_filter_can_skip_an_entire_batch(self):
        model, adapter, _ = build_adapter(threshold=1.0)
        before = model.norm.weight.detach().clone()
        adapter(torch.randn(3, 3))
        self.assertTrue(torch.equal(before, model.norm.weight.detach()))
        self.assertEqual(adapter.stats["adapted_samples"], 0)
        self.assertEqual(adapter.stats["optimizer_steps"], 0)

    def test_subprototype_aggregation(self):
        _, adapter, _ = build_adapter()
        scores = torch.tensor([[[0.1, 0.7], [0.2, 0.5]]])
        aggregated = adapter._aggregate(scores)
        self.assertTrue(torch.equal(aggregated, torch.tensor([[0.7, 0.5]])))

    def test_distance_mapping_matches_log_inverse_kernel(self):
        model = TinyPrototypeModel()
        parameters, _ = configure_model(model)
        adapter = ProtoTTAAdapter(
            model,
            torch.optim.SGD(parameters, lr=0.01),
            prototype_class_identity=model.identity,
            config=ProtoTTAConfig(score_mapping="distance"),
        )
        mapped = adapter._map_scores(torch.tensor([[0.0, 1.0, 10.0]]))
        self.assertGreater(mapped[0, 0], mapped[0, 1])
        self.assertGreater(mapped[0, 1], mapped[0, 2])
        self.assertAlmostEqual(mapped[0, 1].item(), 0.0770, places=3)

    def test_fixed_mode_uses_paper_lambda_convention(self):
        config = ProtoTTAConfig(
            interpolation_mode="fixed",
            fixed_lambda=0.7,
            geometric_threshold=0.0,
        )
        _, adapter, _ = build_adapter(config=config)
        adapter(torch.randn(4, 3))
        self.assertAlmostEqual(adapter.stats["last_lambda_mean"], 0.7, places=6)

    def test_fixed_mode_interpolates_prototype_then_output_loss(self):
        config = ProtoTTAConfig(
            interpolation_mode="fixed",
            fixed_lambda=0.7,
            gradient_normalize=False,
            use_confidence_weighting=False,
            geometric_threshold=0.0,
        )
        model, adapter, _ = build_adapter(config=config)
        inputs = torch.tensor([
            [0.2, 1.0, -0.5],
            [1.2, -0.3, 0.1],
            [-0.4, 0.7, 0.2],
            [0.8, -0.1, -0.6],
        ])
        output = model(inputs)
        batch = adapter.output_extractor(output)
        probabilities = adapter._map_scores(
            adapter._aggregate(batch.prototype_scores)
        )
        prototype_loss, _ = adapter._prototype_loss_per_sample(
            batch, probabilities
        )
        output_probabilities = batch.logits.softmax(dim=1)
        output_loss = -(
            output_probabilities
            * output_probabilities.clamp_min(config.eps).log()
        ).sum(dim=1)
        expected = (0.7 * prototype_loss + 0.3 * output_loss).mean()

        adapter(inputs)
        self.assertAlmostEqual(
            adapter.stats["last_loss"], expected.item(), places=6
        )

    def test_legacy_output_fraction_maps_to_fixed_paper_lambda(self):
        config = ProtoTTAConfig(output_entropy_weight=0.2)
        self.assertEqual(config.interpolation_mode, "fixed")
        self.assertAlmostEqual(config.fixed_lambda, 0.8)

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

    def test_adaptive_router_selects_activation_margin_when_consistent(self):
        config = ProtoTTAConfig(
            interpolation_mode="adaptive",
            use_coverage_guard=False,
            geometric_threshold=0.0,
        )
        model, adapter, _ = build_adapter(config=config)
        activations = torch.tensor([
            [0.9, 0.2], [0.8, 0.3], [0.7, 0.4], [0.95, 0.1]
        ])
        batch = PrototypeBatch(
            logits=torch.tensor([
                [2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]
            ]),
            prototype_scores=activations,
        )
        eligible = torch.tensor([
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]
        ])
        sample_loss = torch.ones(4, requires_grad=True)
        reliable = torch.ones(4, dtype=torch.bool)
        confidence = torch.ones(4)
        with patch.object(
            adapter,
            "_gradient_consistency",
            side_effect=[torch.tensor(0.5), torch.tensor(0.1)],
        ):
            coefficient, diagnostics = adapter._adaptive_lambda(
                batch,
                activations,
                eligible,
                sample_loss,
                sample_loss,
                confidence,
                reliable,
                [model.norm.weight],
            )
        expected = adapter._activation_lambda(batch, activations, eligible)
        self.assertTrue(torch.allclose(coefficient, expected))
        self.assertEqual(diagnostics["router_gate"].item(), 1.0)

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

    def test_coverage_guard_can_select_activation_margin(self):
        config = ProtoTTAConfig(
            interpolation_mode="adaptive",
            use_coverage_guard=True,
            coverage_output_margin=0.9,
            coverage_max_fraction=0.4,
        )
        model, adapter, _ = build_adapter(config=config)
        activations = torch.tensor([
            [0.9, 0.2], [0.8, 0.3], [0.7, 0.4], [0.95, 0.1]
        ])
        batch = PrototypeBatch(
            logits=torch.tensor([
                [10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0]
            ]),
            prototype_scores=activations,
        )
        eligible = torch.tensor([
            [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]
        ])
        reliable = torch.tensor([True, False, False, False])
        sample_loss = torch.ones(4, requires_grad=True)
        with patch.object(
            adapter,
            "_gradient_consistency",
            side_effect=[torch.tensor(0.0), torch.tensor(0.0)],
        ):
            coefficient, diagnostics = adapter._adaptive_lambda(
                batch,
                activations,
                eligible,
                sample_loss,
                sample_loss,
                torch.ones(4),
                reliable,
                [model.norm.weight],
            )
        expected = adapter._activation_lambda(batch, activations, eligible)
        self.assertTrue(torch.allclose(coefficient, expected))
        self.assertEqual(diagnostics["router_gate"].item(), 1.0)
        self.assertAlmostEqual(diagnostics["coverage"].item(), 0.25)

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
        loss, eligible = adapter._prototype_loss_per_sample(
            batch, probabilities
        )
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
        self.assertTrue(torch.equal(eligible, torch.tensor([[1.0, 0.0]])))

    def test_adaptive_forward_records_controller_diagnostics(self):
        _, adapter, _ = build_adapter()
        adapter(torch.randn(6, 3))
        self.assertIsNotNone(adapter.stats["last_activation_lambda_mean"])
        self.assertIsNotNone(adapter.stats["last_relative_lambda_mean"])
        self.assertIn(adapter.stats["last_router_gate"], {0, 1})
        self.assertGreaterEqual(adapter.stats["last_lambda_mean"], 0.0)
        self.assertLessEqual(adapter.stats["last_lambda_mean"], 1.0)

    def test_per_batch_target_mask_needs_no_static_routing(self):
        model = TinyPrototypeModel()
        parameters, _ = configure_model(model)
        optimizer = torch.optim.SGD(parameters, lr=0.01)

        def extract_with_mask(output):
            return {
                **output,
                "target_mask": torch.ones_like(output["prototype_scores"]),
            }

        from ProtoTTA import default_output_extractor

        adapter = ProtoTTAAdapter(
            model,
            optimizer,
            output_extractor=lambda output: default_output_extractor(
                extract_with_mask(output)
            ),
            config=ProtoTTAConfig(
                interpolation_mode="fixed",
                geometric_threshold=0.0,
            ),
        )
        adapter(torch.randn(2, 3))
        self.assertEqual(adapter.stats["optimizer_steps"], 1)


if __name__ == "__main__":
    unittest.main()
