#!/usr/bin/env python3
"""
Enhanced Prototype-based metrics for Test-Time Adaptation (TTA).
Adapted from ProtoViT for ProtoPNet.

New metrics added:
1. PCA-Weighted: Weights by both activation AND class importance (last layer weights)
2. Calibration Score: Similarity of predictions to clean model
3. Class Contribution Change: How ground-truth class prototype contribution changed
4. Adaptation Rate: % of samples actually adapted (for filtering methods)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from .prototype_metrics import PrototypeMetricsEvaluator


class EnhancedPrototypeMetrics(PrototypeMetricsEvaluator):
    """Extended evaluator with additional metrics for ProtoPNet."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        super().__init__(model, device)

        # Extract last layer weights for importance weighting
        # Get core model
        if hasattr(model, 'core'):
            core = model.core
        elif hasattr(model, 'model'):
            if hasattr(model.model, 'core'):
                core = model.model.core
            else:
                core = model.model
        else:
            core = model

        if hasattr(core, 'last_layer'):
            # Shape: [num_classes, num_prototypes]
            self.last_layer_weights = core.last_layer.weight.data.clone()
        else:
            self.last_layer_weights = None
            print("⚠ Warning: Could not extract last layer weights")

        # Storage for clean baseline predictions
        self.clean_predictions = None
        self.clean_logits = None

    def collect_clean_baseline_enhanced(self, clean_loader, max_samples: Optional[int] = None,
                                       verbose: bool = True):
        """
        Collect enhanced clean baseline including predictions and logits.
        """
        # First collect standard baseline
        self.collect_clean_baseline(clean_loader, max_samples, verbose)

        # Now collect predictions and logits
        self.ppnet.eval()
        all_preds = []
        all_logits = []

        with torch.no_grad():
            sample_count = 0
            for images, labels in clean_loader:
                if max_samples and sample_count >= max_samples:
                    break

                images = images.to(self.device)
                outputs = self.ppnet(images)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                all_logits.append(logits.cpu())
                _, preds = logits.max(1)
                all_preds.append(preds.cpu())

                sample_count += images.size(0)

        self.clean_predictions = torch.cat(all_preds)
        self.clean_logits = torch.cat(all_logits)

        if max_samples:
            self.clean_predictions = self.clean_predictions[:max_samples]
            self.clean_logits = self.clean_logits[:max_samples]

        if verbose:
            print(f"✓ Collected clean predictions for {len(self.clean_predictions)} samples")

    def compute_pca_weighted_by_importance(
        self,
        prototype_activations: torch.Tensor,
        labels: torch.Tensor,
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        NEW METRIC: PCA weighted by BOTH activation strength AND class importance.

        This measures how much the activated prototypes actually CONTRIBUTE to the
        prediction of the correct class (not just if they belong to that class).
        """
        if self.last_layer_weights is None:
            return {'PCA_weighted_mean': 0.0, 'PCA_weighted_std': 0.0}

        n_samples = prototype_activations.shape[0]
        weighted_alignment_scores = []

        # Move to CPU for computation
        proto_identities_cpu = self.proto_identities.cpu()
        last_layer_cpu = self.last_layer_weights.cpu()  # [C, P]

        for i in range(n_samples):
            activations = prototype_activations[i]  # [P]
            true_label = labels[i].item()

            # Get top-k activated prototypes
            top_k_values, top_k_indices = torch.topk(activations, k=min(top_k, len(activations)))

            # Get class importance weights for the true class
            importance_weights = last_layer_cpu[true_label, top_k_indices]  # [k]

            # Combine: activation strength * class importance
            combined_contribution = top_k_values * torch.abs(importance_weights)

            # Normalize to get proportion
            total_contribution = combined_contribution.sum()

            # Check if top-k prototypes belong to the true class
            top_k_proto_classes = proto_identities_cpu[top_k_indices]
            correct_class_mask = (top_k_proto_classes == true_label).float()

            # Weighted alignment: sum of contributions from correct-class prototypes
            if total_contribution > 0:
                weighted_alignment = (combined_contribution * correct_class_mask).sum() / total_contribution
            else:
                weighted_alignment = 0.0

            weighted_alignment_scores.append(weighted_alignment.item())

        weighted_alignment_scores = np.array(weighted_alignment_scores)

        return {
            'PCA_weighted_mean': float(np.mean(weighted_alignment_scores)),
            'PCA_weighted_std': float(np.std(weighted_alignment_scores)),
        }

    def compute_calibration_score(
        self,
        adapted_model: nn.Module,
        test_loader,
        max_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """
        NEW METRIC: Calibration Score - How similar are predictions to clean model?

        Measures:
        1. Prediction agreement: % of samples with same predicted class
        2. Logit correlation: Correlation between logit vectors
        """
        if self.clean_predictions is None or self.clean_logits is None:
            return {}

        all_adapted_preds = []
        all_adapted_logits = []

        with torch.no_grad():
            sample_count = 0
            for images, labels in test_loader:
                if max_samples and sample_count >= max_samples:
                    break

                images = images.to(self.device)
                outputs = self._forward_no_adapt(adapted_model, images)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                all_adapted_logits.append(logits.cpu())
                _, preds = logits.max(1)
                all_adapted_preds.append(preds.cpu())

                sample_count += images.size(0)

        adapted_preds = torch.cat(all_adapted_preds)
        adapted_logits = torch.cat(all_adapted_logits)

        if len(adapted_preds) != len(self.clean_predictions):
            raise ValueError(
                'Calibration pairing mismatch: '
                f'adapted={len(adapted_preds)}, '
                f'clean={len(self.clean_predictions)}'
            )
        clean_preds = self.clean_predictions
        clean_logits = self.clean_logits

        # 1. Prediction agreement
        agreement = (adapted_preds == clean_preds).float().mean().item()

        # 2. Logit correlation (per-sample cosine similarity)
        logit_correlations = F.cosine_similarity(adapted_logits, clean_logits, dim=1)
        logit_corr_mean = logit_correlations.mean().item()

        return {
            'calibration_agreement': agreement,  # % same predicted class
            'calibration_logit_corr': logit_corr_mean,  # Logit similarity
        }

    def compute_class_contribution_change(
        self,
        clean_activations: torch.Tensor,
        adapted_activations: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        NEW METRIC: How ground-truth class prototype contribution changed.

        Measures the total contribution of ground-truth class prototypes
        (weighted by their importance) and compares clean vs adapted.
        """
        if self.last_layer_weights is None:
            return {}

        n_samples = len(labels)
        contribution_changes = []
        clean_contributions = []
        adapted_contributions = []

        last_layer_cpu = self.last_layer_weights.cpu()
        proto_identities_cpu = self.proto_identities.cpu()

        for i in range(n_samples):
            true_label = labels[i].item()

            # Get ground-truth class prototypes
            gt_class_mask = (proto_identities_cpu == true_label).float()  # [P]

            # Get importance weights for this class
            importance = torch.abs(last_layer_cpu[true_label, :])  # [P]

            # Compute weighted contribution from GT class prototypes
            clean_contrib = (clean_activations[i] * gt_class_mask * importance).sum().item()
            adapted_contrib = (adapted_activations[i] * gt_class_mask * importance).sum().item()

            # Relative change
            if clean_contrib > 0:
                change = (adapted_contrib - clean_contrib) / clean_contrib
            else:
                change = 0.0

            contribution_changes.append(change)
            clean_contributions.append(clean_contrib)
            adapted_contributions.append(adapted_contrib)

        contribution_changes = np.array(contribution_changes)

        return {
            'gt_class_contrib_change_mean': float(np.mean(contribution_changes)),
            'gt_class_contrib_improvement': float(np.mean(adapted_contributions) - np.mean(clean_contributions)),
        }

    def evaluate_tta_method_enhanced(
        self,
        adapted_model: nn.Module,
        test_loader,
        top_k: int = 10,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Evaluate TTA method with ALL metrics (standard + enhanced).
        """
        # Get standard metrics first
        standard_metrics = self.evaluate_tta_method(
            adapted_model, test_loader, top_k, max_samples, verbose=False
        )

        # Extract activations
        adapted_activations, labels = self.extract_prototype_activations(
            adapted_model, test_loader, max_samples, verbose=False
        )

        # Compute enhanced metrics
        enhanced_metrics = {}

        # 1. PCA weighted by importance
        if verbose:
            print("  Computing PCA weighted by class importance...")
        pca_weighted = self.compute_pca_weighted_by_importance(
            adapted_activations, labels, top_k
        )
        enhanced_metrics.update(pca_weighted)

        # 2. Calibration score
        if self.clean_logits is not None:
            if verbose:
                print("  Computing calibration score...")
            calibration = self.compute_calibration_score(
                adapted_model, test_loader, max_samples
            )
            enhanced_metrics.update(calibration)

        # 3. Class contribution change
        if self.clean_prototype_activations is not None:
            if verbose:
                print("  Computing class contribution change...")
            if len(adapted_activations) != len(self.clean_prototype_activations):
                raise ValueError(
                    'Contribution pairing mismatch: '
                    f'adapted={len(adapted_activations)}, '
                    f'clean={len(self.clean_prototype_activations)}'
                )
            contrib_change = self.compute_class_contribution_change(
                self.clean_prototype_activations,
                adapted_activations,
                labels
            )
            enhanced_metrics.update(contrib_change)

        # Merge all metrics
        all_metrics = {**standard_metrics, **enhanced_metrics}

        if verbose:
            print(f"  PAC (Consistency): {all_metrics.get('PAC_mean', 0)*100:.2f}%")
            print(f"  PCA (Alignment): {all_metrics.get('PCA_mean', 0)*100:.2f}%")
            print(f"  PCA-Weighted: {all_metrics.get('PCA_weighted_mean', 0)*100:.2f}%")
            if 'calibration_agreement' in all_metrics:
                print(f"  Calibration Agreement: {all_metrics.get('calibration_agreement', 0)*100:.1f}%")

        return all_metrics
