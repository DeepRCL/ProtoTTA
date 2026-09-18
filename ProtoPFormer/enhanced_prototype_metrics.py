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
        labels_list = []
        n_samples = 0
        actual = self.ppnet
        actual.eval()
        with torch.no_grad():
            for images, labels in clean_loader:
                if max_samples is not None and n_samples >= max_samples:
                    break
                images = images.to(self.device)
                outputs = actual(images)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                logits_list.append(logits.cpu())
                preds.append(logits.argmax(dim=1).cpu())
                labels_list.append(labels.cpu())
                n_samples += images.size(0)

        self.clean_predictions = torch.cat(preds)
        self.clean_logits = torch.cat(logits_list)
        if max_samples is not None:
            self.clean_predictions = self.clean_predictions[:max_samples]
            self.clean_logits = self.clean_logits[:max_samples]
        clean_labels = torch.cat(labels_list)
        if max_samples is not None:
            clean_labels = clean_labels[:max_samples]
        if not torch.equal(clean_labels, self.clean_labels.cpu()):
            raise RuntimeError("Clean prediction and prototype passes used different sample ordering")
        self.clean_accuracy = float(
            (self.clean_predictions == clean_labels).float().mean().item()
        )

    def evaluate_collected_outputs(
        self,
        activations: torch.Tensor,
        logits: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Compute every Table-3 metric from the exact online prediction pass."""
        lengths = {
            'activations': len(activations), 'logits': len(logits),
            'predictions': len(predictions), 'labels': len(labels),
        }
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Mismatched collected output lengths: {lengths}")
        if self.clean_predictions is None or self.clean_logits is None:
            raise RuntimeError("Clean prediction reference was not collected")
        if self.clean_prototype_activations is None or self.clean_labels is None:
            raise RuntimeError("Clean prototype reference was not collected")

        n = len(labels)
        clean_lengths = {
            'clean activations': len(self.clean_prototype_activations),
            'clean logits': len(self.clean_logits),
            'clean predictions': len(self.clean_predictions),
            'clean labels': len(self.clean_labels),
        }
        if any(value != n for value in clean_lengths.values()):
            raise RuntimeError(
                f"Clean/corrupted pairing requires exactly {n} samples; got {clean_lengths}"
            )

        activations = activations.cpu()
        logits = logits.cpu()
        predictions = predictions.cpu()
        labels = labels.cpu()
        if not torch.equal(labels, self.clean_labels.cpu()):
            raise RuntimeError("Clean/corrupted labels or sample ordering do not match")

        metrics = {}
        metrics.update(self.compute_prototype_activation_consistency(activations))
        metrics.update(self.compute_prototype_class_alignment(activations, labels, top_k=top_k))
        metrics.update(self.compute_prototype_activation_sparsity(activations))
        metrics.update(self.compute_pca_weighted_by_importance(activations, labels, top_k=top_k))

        clean_predictions = self.clean_predictions.cpu()
        clean_logits = self.clean_logits.cpu()
        agreement = float((predictions == clean_predictions).float().mean().item())
        logit_corr = float(F.cosine_similarity(logits, clean_logits, dim=1).mean().item())
        clean_confs = F.softmax(clean_logits, dim=1).max(dim=1)[0]
        corrupt_confs = F.softmax(logits, dim=1).max(dim=1)[0]
        conf_corr = 0.0
        if torch.std(clean_confs) > 1e-6 and torch.std(corrupt_confs) > 1e-6:
            conf_corr = float(np.corrcoef(clean_confs.numpy(), corrupt_confs.numpy())[0, 1])
        metrics.update({
            'calibration_agreement': agreement,
            'calibration_logit_corr': logit_corr,
            'calibration_conf_corr': conf_corr,
        })
        metrics.update(self.compute_class_contribution_change(
            self.clean_prototype_activations.cpu(), activations, labels
        ))

        clean_accuracy = float((clean_predictions == labels).float().mean().item())
        corrupt_accuracy = float((predictions == labels).float().mean().item())
        lower = max(0.0, clean_accuracy + corrupt_accuracy - 1.0)
        upper = 1.0 - abs(clean_accuracy - corrupt_accuracy)
        tolerance = 1.0 / max(n, 1) + 1e-7
        if agreement < lower - tolerance or agreement > upper + tolerance:
            raise RuntimeError(
                "Prediction stability violates paired-accuracy bounds: "
                f"clean={clean_accuracy:.6f}, corrupt={corrupt_accuracy:.6f}, "
                f"stability={agreement:.6f}, allowed=[{lower:.6f}, {upper:.6f}]"
            )
        metrics.update({
            'paired_num_samples': n,
            'clean_accuracy_reference': clean_accuracy,
            'stability_bound_lower': lower,
            'stability_bound_upper': upper,
            'stability_bounds_passed': True,
        })
        return metrics

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
