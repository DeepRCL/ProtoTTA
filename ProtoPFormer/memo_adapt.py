"""MEMO adaptation for ProtoPFormer.

This is the ProtoPFormer-compatible counterpart of ``ProtoViT/memo_adapt.py``.
It preserves the baseline configuration used by the ProtoViT experiments:
episodic per-sample adaptation, 16 AugMix views, all trainable parameters,
one SGD step, and marginal-entropy minimization.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import torchvision.transforms as transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_logits(output):
    return output[0] if isinstance(output, tuple) else output


def marginal_entropy(logits):
    """Entropy of the mean predictive distribution across augmented views."""
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
    avg_log_probs = log_probs.logsumexp(dim=0) - np.log(log_probs.shape[0])
    avg_log_probs = torch.clamp(avg_log_probs, min=torch.finfo(avg_log_probs.dtype).min)
    return -(avg_log_probs * torch.exp(avg_log_probs)).sum(dim=-1)


class MEMO(nn.Module):
    """Episodic MEMO with the same settings as the ProtoViT baseline."""

    def __init__(self, model, optimizer, steps=1, batch_size=16, episodic=True):
        super().__init__()
        if steps < 1:
            raise ValueError("MEMO requires at least one adaptation step")
        if batch_size < 1:
            raise ValueError("MEMO requires at least one augmented view")
        if not episodic:
            raise ValueError("This MEMO baseline is defined as episodic")

        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.batch_size = batch_size
        self.episodic = episodic
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())
        self.aug_fn = create_augmix_augmentation()
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'total_updates': 0,
            'num_augmented_views': batch_size,
        }

    def forward(self, x):
        if x.shape[0] != 1:
            raise ValueError(
                f"MEMO processes one image at a time, received batch size {x.shape[0]}"
            )
        self.reset()
        self.adaptation_stats['total_samples'] += 1

        with torch.enable_grad():
            self.model.train()
            for _ in range(self.steps):
                augmented = self._generate_augmented_views(x)
                self.optimizer.zero_grad()
                logits = _get_logits(self.model(augmented))
                loss = marginal_entropy(logits)
                loss.backward()
                self.optimizer.step()
                self.adaptation_stats['total_updates'] += 1

        self.adaptation_stats['adapted_samples'] += 1
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def _generate_augmented_views(self, x):
        image = x[0]
        mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
        denormalized = image * std + mean
        pil_image = transforms.ToPILImage()(denormalized.cpu())

        augmented = []
        for _ in range(self.batch_size):
            view = transforms.ToTensor()(self.aug_fn(pil_image))
            augmented.append((view - mean.cpu()) / std.cpu())
        return torch.stack(augmented, dim=0).to(x.device, non_blocking=True)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.optimizer.zero_grad()

    def forward_no_adapt(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def create_augmix_augmentation():
    augmentations = [
        autocontrast,
        equalize,
        lambda image: rotate(image, 1),
        lambda image: solarize(image, 1),
        lambda image: posterize(image, 1),
    ]

    def augment(pil_image):
        if np.random.rand() > 0.5:
            pil_image = transforms.RandomResizedCrop(224, scale=(0.8, 1.0))(pil_image)
        else:
            pil_image = transforms.Resize(256)(pil_image)
            pil_image = transforms.RandomCrop(224)(pil_image)
        if np.random.rand() > 0.5:
            pil_image = transforms.RandomHorizontalFlip(p=1.0)(pil_image)
        for _ in range(np.random.randint(1, 4)):
            pil_image = np.random.choice(augmentations)(pil_image)
        return pil_image

    return augment


def autocontrast(image, level=None):
    return ImageOps.autocontrast(image)


def equalize(image, level=None):
    return ImageOps.equalize(image)


def rand_lvl(level):
    return np.random.uniform(low=0.1, high=level)


def int_parameter(level, max_value):
    return int(level * max_value / 10)


def rotate(image, level):
    degrees = int_parameter(rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return image.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)


def solarize(image, level):
    threshold = int_parameter(rand_lvl(level), 256)
    return ImageOps.solarize(image, 256 - threshold)


def posterize(image, level):
    bits = int_parameter(rand_lvl(level), 4)
    return ImageOps.posterize(image, 4 - bits)


def setup_memo(model, lr=0.00025, batch_size=16, steps=1):
    """Configure the all-parameter episodic MEMO baseline."""
    model.train()
    model.requires_grad_(True)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=0.9, weight_decay=0.0
    )
    return MEMO(
        model,
        optimizer,
        steps=steps,
        batch_size=batch_size,
        episodic=True,
    )
