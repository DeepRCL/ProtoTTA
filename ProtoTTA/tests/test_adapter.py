import unittest

import torch
from torch import nn

from ProtoTTA import ProtoTTAAdapter, ProtoTTAConfig, configure_model


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


def build_adapter(*, threshold=0.0):
    model = TinyPrototypeModel()
    parameters, names = configure_model(model)
    optimizer = torch.optim.SGD(parameters, lr=0.05)
    adapter = ProtoTTAAdapter(
        model,
        optimizer,
        prototype_class_identity=model.identity,
        config=ProtoTTAConfig(geometric_threshold=threshold),
    )
    return model, adapter, names


class ProtoTTAAdapterTest(unittest.TestCase):
    def test_configure_model_selects_only_normalization(self):
        model, _, names = build_adapter()
        self.assertEqual(names, ["norm.weight", "norm.bias"])
        self.assertFalse(model.prototypes.requires_grad)

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
            config=ProtoTTAConfig(geometric_threshold=0.0),
        )
        adapter(torch.randn(2, 3))
        self.assertEqual(adapter.stats["optimizer_steps"], 1)


if __name__ == "__main__":
    unittest.main()
