#!/usr/bin/env python3
"""Enhanced prototype metrics for ProtoPFormer TTA."""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prototype_tta_metrics import PrototypeMetricsEvaluator


class EnhancedPrototypeMetrics(PrototypeMetricsEvaluator):
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        super().__init__(model, device)

        self.clean_predictions = None
        self.clean_logits = None

        weights = [(1.0 - float(getattr(self.ppnet, 'global_coe', 0.5))) * self.ppnet.last_layer.weight.data.clone()]
        if hasattr(self.ppnet, 'last_layer_global'):
            weights.append(float(getattr(self.ppnet, 'global_coe', 0.5)) * self.ppnet.last_layer_global.weight.data.clone())
        self.last_layer_weights = torch.cat(weights, dim=1).cpu()

    def collect_clean_baseline_enhanced(self, clean_loader, max_samples: Optional[int] = None, verbose: bool = True):
        self.collect_clean_baseline(clean_loader, max_samples=max_samples, verbose=verbose)

        preds = []
        logits_list = []
        n_samples = 0
        actual = self.ppnet
        actual.eval()
        with torch.no_grad():
            for images, _ in clean_loader:
                if max_samples is not None and n_samples >= max_samples:
                    break
                images = images.to(self.device)
                outputs = actual(images)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                logits_list.append(logits.cpu())
                preds.append(logits.argmax(dim=1).cpu())
                n_samples += images.size(0)

        self.clean_predictions = torch.cat(preds)
        self.clean_logits = torch.cat(logits_list)
        if max_samples is not None:
            self.clean_predictions = self.clean_predictions[:max_samples]
            self.clean_logits = self.clean_logits[:max_samples]

    def compute_pca_weighted_by_importance(self, activations: torch.Tensor, labels: torch.Tensor, top_k: int = 10) -> Dict[str, float]:
        proto_ids = self.proto_identities.cpu()
        weighted_scores = []
        for sample_acts, label in zip(activations, labels):
            top_vals, top_idx = torch.topk(sample_acts, k=min(top_k, sample_acts.numel()))
            importance = torch.abs(self.last_layer_weights[label.item(), top_idx])
            contrib = top_vals * importance
            total = contrib.sum()
            correct = (proto_ids[top_idx] == label.item()).float()
            score = float(((contrib * correct).sum() / total).item()) if total > 0 else 0.0
            weighted_scores.append(score)
        weighted_scores = np.array(weighted_scores)
        return {
            'PCA_weighted_mean': float(np.mean(weighted_scores)),
            'PCA_weighted_std': float(np.std(weighted_scores)),
        }

    def compute_calibration_score(self, adapted_model: nn.Module, test_loader, max_samples: Optional[int] = None) -> Dict[str, float]:
        # Prediction Stability is defined against the clean model on clean data
        # versus the current method on noisy data for the corresponding test images.
        if self.clean_predictions is None or self.clean_logits is None:
            return {}

        adapted_preds_list = []
        adapted_logits_list = []

        with torch.no_grad():
            for images, _ in test_loader:
                if max_samples is not None and len(adapted_preds_list) > 0:
                    collected = sum(x.shape[0] for x in adapted_preds_list)
                    if collected >= max_samples:
                        break
                images = images.to(self.device)
                adapted_out = self._forward_no_adapt(adapted_model, images)
                adapted_logits = adapted_out[0] if isinstance(adapted_out, tuple) else adapted_out
                adapted_preds_list.append(adapted_logits.argmax(dim=1).cpu())
                adapted_logits_list.append(adapted_logits.cpu())

        adapted_preds = torch.cat(adapted_preds_list)
        adapted_logits_t = torch.cat(adapted_logits_list)
        if max_samples is not None:
            adapted_preds = adapted_preds[:max_samples]
            adapted_logits_t = adapted_logits_t[:max_samples]

        n = min(len(adapted_preds), len(self.clean_predictions))
        clean_preds = self.clean_predictions[:n]
        clean_logits_t = self.clean_logits[:n]
        adapted_preds = adapted_preds[:n]
        adapted_logits_t = adapted_logits_t[:n]

        agreement = (adapted_preds == clean_preds).float().mean().item()
        logit_corr = F.cosine_similarity(adapted_logits_t, clean_logits_t, dim=1).mean().item()

        clean_confs = F.softmax(clean_logits_t, dim=1).max(dim=1)[0]
        adapted_confs = F.softmax(adapted_logits_t, dim=1).max(dim=1)[0]
        if torch.std(clean_confs) > 1e-6 and torch.std(adapted_confs) > 1e-6:
            conf_corr = float(np.corrcoef(clean_confs.numpy(), adapted_confs.numpy())[0, 1])
        else:
            conf_corr = 0.0

        return {
            'calibration_agreement': float(agreement),
            'calibration_logit_corr': float(logit_corr),
            'calibration_conf_corr': float(conf_corr),
        }

    def compute_class_contribution_change(self, clean_activations: torch.Tensor, adapted_activations: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        proto_ids = self.proto_identities.cpu()
        changes = []
        clean_vals = []
        adapted_vals = []
        for clean_vec, adapt_vec, label in zip(clean_activations, adapted_activations, labels):
            class_mask = (proto_ids == label.item()).float()
            importance = torch.abs(self.last_layer_weights[label.item()])
            clean_contrib = float((clean_vec * class_mask * importance).sum().item())
            adapt_contrib = float((adapt_vec * class_mask * importance).sum().item())
            delta = (adapt_contrib - clean_contrib) / clean_contrib if clean_contrib > 0 else 0.0
            changes.append(delta)
            clean_vals.append(clean_contrib)
            adapted_vals.append(adapt_contrib)
        return {
            'gt_class_contrib_change_mean': float(np.mean(changes)),
            'gt_class_contrib_change_std': float(np.std(changes)),
            'gt_class_contrib_improvement': float(np.mean(adapted_vals) - np.mean(clean_vals)),
        }

    def evaluate_tta_method_enhanced(
        self,
        adapted_model: nn.Module,
        test_loader,
        top_k: int = 10,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        track_adaptation_rate: bool = False,
    ) -> Dict[str, float]:
        metrics = self.evaluate_tta_method(adapted_model, test_loader, top_k=top_k, max_samples=max_samples, verbose=verbose)
        activations, labels = self.extract_prototype_activations(adapted_model, test_loader, max_samples=max_samples, verbose=False)

        metrics.update(self.compute_pca_weighted_by_importance(activations, labels, top_k=top_k))
        metrics.update(self.compute_calibration_score(adapted_model, test_loader, max_samples=max_samples))

        if self.clean_prototype_activations is not None:
            n = min(len(activations), len(self.clean_prototype_activations))
            metrics.update(self.compute_class_contribution_change(
                self.clean_prototype_activations[:n], activations[:n], labels[:n]
            ))

        if track_adaptation_rate and hasattr(adapted_model, 'adaptation_stats'):
            stats = adapted_model.adaptation_stats
            total = max(stats.get('total_samples', 1), 1)
            metrics['adaptation_rate'] = stats.get('adapted_samples', 0) / total
            metrics['avg_updates_per_sample'] = stats.get('total_updates', 0) / total

        return metrics
