#!/usr/bin/env python3
"""Prototype-based TTA metrics for ProtoPFormer."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMetricsEvaluator:
    """Evaluate PAC, PCA, and sparsity for ProtoPFormer TTA methods."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.device = device
        self.ppnet = model.model if hasattr(model, 'model') else model

        local_identity = self.ppnet.prototype_class_identity.to(device)
        self.local_proto_identities = local_identity.argmax(dim=1)
        self.num_local = local_identity.shape[0]

        self.global_proto_identities = None
        self.num_global = 0
        if hasattr(self.ppnet, 'prototype_class_identity_global'):
            global_identity = self.ppnet.prototype_class_identity_global.to(device)
            self.global_proto_identities = global_identity.argmax(dim=1)
            self.num_global = global_identity.shape[0]

        if self.global_proto_identities is not None:
            self.proto_identities = torch.cat(
                [self.local_proto_identities, self.global_proto_identities], dim=0
            )
        else:
            self.proto_identities = self.local_proto_identities

        self.clean_prototype_activations = None
        self.clean_labels = None

    def _forward_no_adapt(self, model: nn.Module, images: torch.Tensor):
        if hasattr(model, 'forward_no_adapt'):
            return model.forward_no_adapt(images)
        actual = model.model if hasattr(model, 'model') else model
        return actual(images)

    def _extract_prototype_activations(self, model: nn.Module, outputs) -> torch.Tensor:
        actual = model.model if hasattr(model, 'model') else model

        if not isinstance(outputs, tuple) or len(outputs) < 2:
            raise ValueError("Expected ProtoPFormer output tuple")

        aux = outputs[1]
        if not isinstance(aux, (tuple, list)):
            raise ValueError("Expected auxiliary ProtoPFormer outputs")

        local_acts = None
        global_acts = None

        # Eval mode format:
        #   (cls_token_attn, distances, logits_global, logits_local, local_acts, global_acts)
        if (
            len(aux) >= 6
            and isinstance(aux[4], torch.Tensor)
            and aux[4].dim() == 2
            and isinstance(aux[5], torch.Tensor)
            and aux[5].dim() == 2
        ):
            local_acts = aux[4]
            global_acts = aux[5]
        # Train/TTA mode format:
        #   (student_token_attn, attn_loss, total_proto_act, cls_attn_rollout,
        #    original_fea_len, local_acts, global_acts)
        elif (
            len(aux) >= 7
            and isinstance(aux[5], torch.Tensor)
            and aux[5].dim() == 2
        ):
            local_acts = aux[5]
            if isinstance(aux[6], torch.Tensor) and aux[6].dim() == 2:
                global_acts = aux[6]
        elif len(aux) >= 2 and isinstance(aux[1], torch.Tensor) and aux[1].dim() == 4:
            distances = aux[1]
            local_acts = actual.distance_2_similarity(distances)
            if local_acts.dim() == 4:
                fea_size = local_acts.shape[-1]
                if fea_size > 1:
                    local_acts = F.max_pool2d(local_acts, kernel_size=(fea_size, fea_size))
                local_acts = local_acts.view(local_acts.size(0), -1)

        if local_acts is None:
            raise ValueError("Could not extract local prototype activations")

        if global_acts is not None:
            return torch.cat([local_acts, global_acts], dim=1)
        return local_acts

    def extract_prototype_activations(
        self,
        model: nn.Module,
        loader,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_activations = []
        all_labels = []
        n_samples = 0

        iterator = loader
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="Extracting prototypes")

        with torch.no_grad():
            for images, labels in iterator:
                if max_samples is not None and n_samples >= max_samples:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self._forward_no_adapt(model, images)
                activations = self._extract_prototype_activations(model, outputs)

                all_activations.append(activations.detach().cpu())
                all_labels.append(labels.detach().cpu())
                n_samples += labels.size(0)

        activations = torch.cat(all_activations, dim=0)
        labels = torch.cat(all_labels, dim=0)
        if max_samples is not None:
            activations = activations[:max_samples]
            labels = labels[:max_samples]
        return activations, labels

    def collect_clean_baseline(self, clean_loader, max_samples: Optional[int] = None, verbose: bool = True):
        self.clean_prototype_activations, self.clean_labels = self.extract_prototype_activations(
            self.ppnet, clean_loader, max_samples=max_samples, verbose=verbose
        )

    def compute_prototype_activation_consistency(
        self,
        adapted_activations: torch.Tensor,
        clean_activations: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        clean_activations = clean_activations if clean_activations is not None else self.clean_prototype_activations
        if clean_activations is None:
            raise ValueError("No clean baseline collected")

        n = min(len(adapted_activations), len(clean_activations))
        sims = F.cosine_similarity(
            adapted_activations[:n].float(),
            clean_activations[:n].float(),
            dim=1,
        ).cpu().numpy()
        return {
            'PAC_mean': float(np.mean(sims)),
            'PAC_std': float(np.std(sims)),
            'PAC_median': float(np.median(sims)),
            'PAC_min': float(np.min(sims)),
            'PAC_max': float(np.max(sims)),
        }

    def compute_prototype_class_alignment(
        self,
        prototype_activations: torch.Tensor,
        labels: torch.Tensor,
        top_k: int = 10,
    ) -> Dict[str, float]:
        proto_ids = self.proto_identities.cpu()
        scores = []
        for activations, label in zip(prototype_activations, labels):
            top_vals, top_idx = torch.topk(activations, k=min(top_k, activations.numel()))
            correct = (proto_ids[top_idx] == label.item()).float()
            weights = F.softmax(top_vals.float(), dim=0)
            scores.append(float((correct * weights).sum().item()))
        scores = np.array(scores)
        return {
            'PCA_mean': float(np.mean(scores)),
            'PCA_std': float(np.std(scores)),
            'PCA_median': float(np.median(scores)),
            'PCA_min': float(np.min(scores)),
            'PCA_max': float(np.max(scores)),
        }

    def compute_prototype_activation_sparsity(
        self,
        prototype_activations: torch.Tensor,
        threshold: float = 0.1,
    ) -> Dict[str, float]:
        def gini(array):
            array = np.abs(array)
            if array.sum() == 0:
                return 0.0
            sorted_array = np.sort(array)
            n = len(array)
            cumsum = np.cumsum(sorted_array)
            return (n + 1 - 2 * np.sum(cumsum) / array.sum()) / n

        gini_scores = []
        active_counts = []
        for activations in prototype_activations:
            arr = activations.numpy()
            gini_scores.append(gini(arr))
            active_counts.append(int((np.abs(arr) > threshold).sum()))

        return {
            'sparsity_gini_mean': float(np.mean(gini_scores)),
            'sparsity_gini_std': float(np.std(gini_scores)),
            'sparsity_active_mean': float(np.mean(active_counts)),
            'sparsity_active_std': float(np.std(active_counts)),
        }

    def evaluate_tta_method(
        self,
        adapted_model: nn.Module,
        test_loader,
        top_k: int = 10,
        max_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        adapted_activations, labels = self.extract_prototype_activations(
            adapted_model, test_loader, max_samples=max_samples, verbose=verbose
        )

        metrics = {}
        if self.clean_prototype_activations is not None:
            metrics.update(self.compute_prototype_activation_consistency(adapted_activations))
        metrics.update(self.compute_prototype_class_alignment(adapted_activations, labels, top_k=top_k))
        metrics.update(self.compute_prototype_activation_sparsity(adapted_activations))
        return metrics
