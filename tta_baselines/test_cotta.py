import unittest

import torch
from torch import nn

from tta_baselines import CoTTA, TokenMaskTransform


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 2)

    def forward(self, inputs):
        return self.classifier(inputs)


class CoTTATest(unittest.TestCase):
    def test_forward_tracks_online_statistics(self):
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        identity = lambda args, kwargs: (args, kwargs)
        wrapper = CoTTA(
            model,
            optimizer,
            identity,
            ap=0.0,
            rst_m=0.0,
            n_augmentations=1,
        )

        output = wrapper(torch.ones(4, 3))

        self.assertEqual(output.shape, (4, 2))
        self.assertEqual(wrapper.adaptation_stats["total_samples"], 4)
        self.assertEqual(wrapper.adaptation_stats["total_updates"], 1)
        self.assertIs(wrapper.metric_model, wrapper.model_ema)
        self.assertIs(wrapper.last_metric_output, output)

    def test_uncertain_anchor_uses_augmentation_teacher(self):
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def perturb(args, kwargs):
            return ((args[0] + 0.1,), kwargs)

        wrapper = CoTTA(
            model,
            optimizer,
            perturb,
            ap=1.01,
            rst_m=0.0,
            n_augmentations=2,
        )
        wrapper(torch.ones(2, 3))
        self.assertEqual(
            wrapper.adaptation_stats["augmentation_teacher_batches"], 1
        )

    def test_token_mask_avoids_special_and_padding_tokens(self):
        transform = TokenMaskTransform(mask_token_id=99, probability=1.0)
        arguments, keywords = transform(
            (),
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "special_tokens_mask": torch.tensor([[1, 0, 0, 0]]),
            },
        )
        self.assertEqual(arguments, ())
        self.assertEqual(keywords["input_ids"].tolist(), [[1, 99, 99, 4]])


if __name__ == "__main__":
    unittest.main()
