"""CoTTA baseline shared by the image and text robustness evaluators.

The adaptation rule follows the authors' public implementation:
https://github.com/qinenergy/cotta (MIT license).  The wrapper keeps an EMA
teacher, uses augmentation-averaged teacher predictions when the source anchor
is uncertain, and stochastically restores adapted student parameters.

Unlike an ImageNet-only implementation, this module accepts an augmentation
callable over ``(*args, **kwargs)``.  This makes it possible to use the same CoTTA
algorithm for normalized images and tokenized text without changing the loss or
the online update rule.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as vision_functional
from torchvision.transforms.functional import InterpolationMode


Args = Tuple[Any, ...]
Kwargs = Dict[str, Any]
Augment = Callable[[Args, Kwargs], Tuple[Args, Kwargs]]


def _logits(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _running_output_mean(current: Any, new: Any, count: int) -> Any:
    """Update every tensor leaf without retaining all augmentation outputs."""
    if current is None:
        return new
    if isinstance(current, torch.Tensor):
        return current.add((new - current) / count)
    if isinstance(current, tuple):
        return tuple(_running_output_mean(old, fresh, count)
                     for old, fresh in zip(current, new, strict=True))
    if isinstance(current, list):
        return [_running_output_mean(old, fresh, count)
                for old, fresh in zip(current, new, strict=True)]
    if isinstance(current, dict):
        return {key: _running_output_mean(current[key], new[key], count)
                for key in current}
    return current


def _detach_model(model: nn.Module) -> nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model


def _symmetric_cross_entropy(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """CoTTA's ImageNet loss: CE(student, teacher) + CE(teacher, student)."""
    teacher_prob = teacher.softmax(dim=1)
    student_prob = student.softmax(dim=1)
    return -0.5 * (
        (teacher_prob * student.log_softmax(dim=1)).sum(dim=1)
        + (student_prob * teacher.log_softmax(dim=1)).sum(dim=1)
    ).mean()


def _teacher_cross_entropy(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """CoTTA's CIFAR loss: teacher probabilities supervise the student."""
    return -(teacher.softmax(dim=1) * student.log_softmax(dim=1)).sum(dim=1).mean()


class CoTTA(nn.Module):
    """Continual Test-Time Adaptation with an EMA/augmentation teacher.

    Predictions are returned from the EMA teacher used for that online step
    (augmentation-averaged when the source anchor is uncertain).  Prototype
    activations and logits consumed by the interpretability evaluator therefore
    describe exactly the same prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        augment: Augment,
        *,
        steps: int = 1,
        episodic: bool = False,
        mt_alpha: float = 0.999,
        rst_m: float = 0.001,
        ap: float = 0.1,
        n_augmentations: int = 32,
        symmetric_loss: bool = True,
    ) -> None:
        super().__init__()
        if steps < 1 or n_augmentations < 1:
            raise ValueError("steps and n_augmentations must be positive")
        if not 0.0 <= mt_alpha <= 1.0 or not 0.0 <= rst_m <= 1.0:
            raise ValueError("mt_alpha and rst_m must be in [0, 1]")

        self.model = model
        self.optimizer = optimizer
        self.augment = augment
        self.steps = steps
        self.episodic = episodic
        self.mt_alpha = mt_alpha
        self.rst_m = rst_m
        self.ap = ap
        self.n_augmentations = n_augmentations
        self.symmetric_loss = symmetric_loss

        self.model_ema = _detach_model(deepcopy(model))
        self.model_anchor = _detach_model(deepcopy(model))
        # Evaluators hook this model so returned logits and prototype evidence
        # are guaranteed to correspond to the same teacher forward.
        self.metric_model = self.model_ema

        self._trainable_names = {
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        self._source_params = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
            if name in self._trainable_names
        }
        self._initial_student = deepcopy(self.model.state_dict()) if episodic else None
        self._initial_ema = deepcopy(self.model_ema.state_dict()) if episodic else None
        self._initial_optimizer = deepcopy(self.optimizer.state_dict()) if episodic else None
        self.adaptation_stats = {
            "total_samples": 0,
            "adapted_samples": 0,
            "total_updates": 0,
            "augmentation_teacher_batches": 0,
        }
        self.last_metric_output = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.episodic:
            self.reset()
        batch_size = self._batch_size(args, kwargs)
        self.adaptation_stats["total_samples"] += batch_size
        self.adaptation_stats["adapted_samples"] += batch_size
        output = None
        for _ in range(self.steps):
            output = self.forward_and_adapt(*args, **kwargs)
            self.adaptation_stats["total_updates"] += 1
        return output

    def train(self, mode: bool = True) -> "CoTTA":
        """Keep CoTTA's configured student mode when evaluators call ``eval``."""
        self.training = mode
        if mode:
            self.model.train()
        self.model_ema.eval()
        self.model_anchor.eval()
        return self

    @staticmethod
    def _batch_size(args: Args, kwargs: Kwargs) -> int:
        tensors = [value for value in args if isinstance(value, torch.Tensor)]
        tensors.extend(value for value in kwargs.values() if isinstance(value, torch.Tensor))
        if not tensors:
            raise ValueError("CoTTA requires at least one batched tensor input")
        return int(tensors[0].shape[0])

    @torch.enable_grad()
    def forward_and_adapt(self, *args: Any, **kwargs: Any) -> Any:
        call_args, call_kwargs = tuple(args), dict(kwargs)

        with torch.no_grad():
            anchor_output = self.model_anchor(*call_args, **call_kwargs)
            anchor_confidence = _logits(anchor_output).softmax(dim=1).amax(dim=1).mean()

            if float(anchor_confidence) < self.ap:
                teacher_output = None
                for augmentation_index in range(1, self.n_augmentations + 1):
                    aug_args, aug_kwargs = self.augment(call_args, call_kwargs)
                    augmented_output = self.model_ema(*aug_args, **aug_kwargs)
                    teacher_output = _running_output_mean(
                        teacher_output, augmented_output, augmentation_index
                    )
                # The public CoTTA prediction is the augmentation-averaged EMA
                # output.  Average prototype leaves as well so reported metrics
                # describe the same ensemble evidence as the returned logits.
                self.adaptation_stats["augmentation_teacher_batches"] += 1
            else:
                teacher_output = self.model_ema(*call_args, **call_kwargs)
            teacher_target = _logits(teacher_output)

        self.last_metric_output = teacher_output

        student_output = self.model(*call_args, **call_kwargs)
        loss_fn = _symmetric_cross_entropy if self.symmetric_loss else _teacher_cross_entropy
        loss = loss_fn(_logits(student_output), teacher_target.detach())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self._ema_update()
        self._stochastic_restore()
        return teacher_output

    @torch.no_grad()
    def _ema_update(self) -> None:
        for ema_parameter, parameter in zip(
            self.model_ema.parameters(), self.model.parameters(), strict=True
        ):
            ema_parameter.mul_(self.mt_alpha).add_(parameter.detach(), alpha=1.0 - self.mt_alpha)

    @torch.no_grad()
    def _stochastic_restore(self) -> None:
        if self.rst_m == 0.0:
            return
        for name, parameter in self.model.named_parameters():
            if name not in self._trainable_names:
                continue
            source = self._source_params[name].to(device=parameter.device, dtype=parameter.dtype)
            restore = torch.rand_like(parameter, dtype=torch.float32) < self.rst_m
            parameter.copy_(torch.where(restore, source, parameter))

    def reset(self) -> None:
        if self._initial_student is None:
            raise RuntimeError("reset is only available when episodic=True")
        self.model.load_state_dict(self._initial_student, strict=True)
        self.model_ema.load_state_dict(self._initial_ema, strict=True)
        self.optimizer.load_state_dict(self._initial_optimizer)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            modules = self.__dict__.get("_modules", {})
            metric_model = modules.get("model_ema")
            if metric_model is not None and hasattr(metric_model, name):
                return getattr(metric_model, name)
            raise


class CoTTAImageTransform:
    """CoTTA augmentation for already-normalized image batches."""

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        *,
        image_size: int = 224,
        gaussian_std: float = 0.005,
        soft: bool = False,
    ) -> None:
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
        self.gaussian_std = gaussian_std
        padding = image_size // 2
        self.ranges = {
            "brightness": (0.8, 1.2) if soft else (0.6, 1.4),
            "contrast": (0.85, 1.15) if soft else (0.7, 1.3),
            "saturation": (0.75, 1.25) if soft else (0.5, 1.5),
            "hue": (-0.03, 0.03) if soft else (-0.06, 0.06),
            "gamma": (0.85, 1.15) if soft else (0.7, 1.3),
        }
        self.transform = transforms.Compose([
            transforms.Pad(padding=padding, padding_mode="edge"),
            transforms.RandomAffine(
                degrees=(-8, 8) if soft else (-15, 15),
                translate=(1 / 16, 1 / 16),
                scale=(0.95, 1.05) if soft else (0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.GaussianBlur(
                5, sigma=(0.001, 0.25) if soft else (0.001, 0.5)
            ),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ])

    def _color_jitter_with_gamma(self, image: torch.Tensor) -> torch.Tensor:
        operations = ["brightness", "contrast", "saturation", "hue", "gamma"]
        for index in torch.randperm(len(operations)).tolist():
            operation = operations[index]
            low, high = self.ranges[operation]
            factor = torch.empty((), device=image.device).uniform_(low, high).item()
            if operation == "brightness":
                image = vision_functional.adjust_brightness(image, factor)
            elif operation == "contrast":
                image = vision_functional.adjust_contrast(image, factor)
            elif operation == "saturation":
                image = vision_functional.adjust_saturation(image, factor)
            elif operation == "hue":
                image = vision_functional.adjust_hue(image, factor)
            else:
                image = vision_functional.adjust_gamma(image.clamp(1e-8, 1.0), factor)
        return image

    def __call__(self, args: Args, kwargs: Kwargs) -> Tuple[Args, Kwargs]:
        if not args or not isinstance(args[0], torch.Tensor):
            raise ValueError("Image CoTTA expects the image batch as its first argument")
        x = args[0]
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        pixels = (x * std + mean).clamp(0.0, 1.0)
        pixels = self._color_jitter_with_gamma(pixels)
        pixels = self.transform(pixels)
        pixels = (pixels + torch.randn_like(pixels) * self.gaussian_std).clamp(0.0, 1.0)
        normalized = (pixels - mean) / std
        return (normalized, *args[1:]), dict(kwargs)


class TokenMaskTransform:
    """Text analogue of CoTTA's stochastic input augmentation."""

    def __init__(self, mask_token_id: int, probability: float = 0.10) -> None:
        if mask_token_id is None:
            raise ValueError("Tokenizer must provide a mask token for text CoTTA")
        self.mask_token_id = int(mask_token_id)
        self.probability = probability

    def __call__(self, args: Args, kwargs: Kwargs) -> Tuple[Args, Kwargs]:
        result = dict(kwargs)
        input_ids = result["input_ids"].clone()
        eligible = result.get("attention_mask", torch.ones_like(input_ids)).bool()
        special = result.get("special_tokens_mask")
        if special is not None:
            eligible &= ~special.bool()
        mask = (torch.rand(input_ids.shape, device=input_ids.device) < self.probability) & eligible
        input_ids[mask] = self.mask_token_id
        result["input_ids"] = input_ids
        return tuple(args), result
