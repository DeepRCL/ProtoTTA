#!/usr/bin/env python3
"""
Enhanced Prototype-based metrics for Test-Time Adaptation (TTA).

New metrics added:
1. PCA-Weighted: Weights by both activation AND class importance (last layer weights)
2. Calibration Score: Similarity of predictions to clean model
3. Class Contribution Change: How ground-truth class prototype contribution changed
4. Adaptation Rate: % of samples actually adapted (for filtering methods)
5. Per-Sample Update Count: Track actual gradient updates per sample
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from prototype_tta_metrics import PrototypeMetricsEvaluator


class EnhancedPrototypeMetrics(PrototypeMetricsEvaluator):
    """Extended evaluator with additional metrics."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        super().__init__(model, device)
        
        # Extract last layer weights for importance weighting
        if hasattr(self.ppnet, 'last_layer'):
            # Shape: [num_classes, num_prototypes]
            self.last_layer_weights = self.ppnet.last_layer.weight.data.clone()
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
            for images, labels in clean_loader:
                if max_samples and len(all_preds) >= max_samples:
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
        
        self.clean_predictions = torch.cat(all_preds)
        self.clean_logits = torch.cat(all_logits)
        
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
        
        Args:
            prototype_activations: [N, P]
            labels: [N]
            top_k: Number of top prototypes to consider
        
        Returns:
            Dictionary with weighted PCA metrics
        """
        if self.last_layer_weights is None:
            return {'PCA_weighted_mean': 0.0, 'error': 'No last layer weights'}
        
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
            # last_layer[true_label, proto_idx] = how important proto_idx is for predicting true_label
            importance_weights = last_layer_cpu[true_label, top_k_indices]  # [k]
            
            # Combine: activation strength * class importance
            # This tells us: "how much does this activated prototype contribute to the correct class?"
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
            'PCA_weighted_median': float(np.median(weighted_alignment_scores)),
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
        3. Confidence preservation: How much prediction confidence is maintained
        
        Args:
            adapted_model: Adapted model to evaluate
            test_loader: DataLoader
            max_samples: Max samples to evaluate
        
        Returns:
            Dictionary with calibration metrics
        """
        if self.clean_predictions is None or self.clean_logits is None:
            return {'error': 'No clean baseline. Call collect_clean_baseline_enhanced() first'}
        
        # DON'T call adapted_model.eval() here!
        # Some TTA methods like Tent REQUIRE .train() mode to work properly
        
        all_adapted_preds = []
        all_adapted_logits = []
        all_adapted_confidences = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                if max_samples and len(all_adapted_preds) >= max_samples:
                    break
                
                images = images.to(self.device)
                outputs = self._forward_no_adapt(adapted_model, images)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                probs = F.softmax(logits, dim=1)
                confidences, preds = probs.max(1)
                
                all_adapted_logits.append(logits.cpu())
                all_adapted_preds.append(preds.cpu())
                all_adapted_confidences.append(confidences.cpu())
        
        adapted_preds = torch.cat(all_adapted_preds)
        adapted_logits = torch.cat(all_adapted_logits)
        adapted_confidences = torch.cat(all_adapted_confidences)
        
        # Truncate to same length
        min_len = min(len(adapted_preds), len(self.clean_predictions))
        adapted_preds = adapted_preds[:min_len]
        adapted_logits = adapted_logits[:min_len]
        adapted_confidences = adapted_confidences[:min_len]
        clean_preds = self.clean_predictions[:min_len]
        clean_logits = self.clean_logits[:min_len]
        
        # Compute clean confidences
        clean_probs = F.softmax(clean_logits, dim=1)
        clean_confidences = clean_probs.max(1)[0]
        
        # 1. Prediction agreement
        agreement = (adapted_preds == clean_preds).float().mean().item()
        
        # 2. Logit correlation (per-sample cosine similarity)
        logit_correlations = F.cosine_similarity(adapted_logits, clean_logits, dim=1)
        logit_corr_mean = logit_correlations.mean().item()
        
        # 3. Confidence preservation
        conf_diff = (adapted_confidences - clean_confidences).abs().mean().item()
        conf_corr = np.corrcoef(clean_confidences.numpy(), adapted_confidences.numpy())[0, 1]
        
        return {
            'calibration_agreement': agreement,  # % same predicted class
            'calibration_logit_corr': logit_corr_mean,  # Logit similarity
            'calibration_conf_diff': conf_diff,  # Avg confidence change
            'calibration_conf_corr': conf_corr,  # Confidence correlation
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
        
        Args:
            clean_activations: [N, P] - from clean model
            adapted_activations: [N, P] - from adapted model
            labels: [N] - ground truth labels
        
        Returns:
            Dictionary with contribution change metrics
        """
        if self.last_layer_weights is None:
            return {'error': 'No last layer weights'}
        
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
            'gt_class_contrib_change_std': float(np.std(contribution_changes)),
            'gt_class_contrib_clean_mean': float(np.mean(clean_contributions)),
            'gt_class_contrib_adapted_mean': float(np.mean(adapted_contributions)),
            'gt_class_contrib_improvement': float(np.mean(adapted_contributions) - np.mean(clean_contributions)),
        }
    
    def evaluate_tta_method_enhanced(
        self,
        adapted_model: nn.Module,
        test_loader,
        top_k: int = 10,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        track_adaptation_rate: bool = False
    ) -> Dict[str, any]:
        """
        Evaluate TTA method with ALL metrics (standard + enhanced).
        
        Args:
            adapted_model: TTA-adapted model
            test_loader: Test data loader
            top_k: Number of top prototypes for alignment metrics
            max_samples: Maximum samples to evaluate
            verbose: Print progress
            track_adaptation_rate: Track adaptation statistics
        
        Returns:
            Dictionary with all metrics
        """
        # Get standard metrics first
        standard_metrics = self.evaluate_tta_method(
            adapted_model, test_loader, top_k, max_samples, verbose
        )
        
        # Extract activations
        adapted_activations, labels = self.extract_prototype_activations(
            adapted_model, test_loader, max_samples, verbose=False
        )
        
        # Compute enhanced metrics
        enhanced_metrics = {}
        
        # 1. PCA weighted by importance
        if verbose:
            print("Computing PCA weighted by class importance...")
        pca_weighted = self.compute_pca_weighted_by_importance(
            adapted_activations, labels, top_k
        )
        enhanced_metrics.update(pca_weighted)
        
        # 2. Calibration score
        if self.clean_logits is not None:
            if verbose:
                print("Computing calibration score...")
            calibration = self.compute_calibration_score(
                adapted_model, test_loader, max_samples
            )
            enhanced_metrics.update(calibration)
        
        # 3. Class contribution change
        if self.clean_prototype_activations is not None:
            if verbose:
                print("Computing class contribution change...")
            # Truncate to same length
            min_len = min(len(adapted_activations), len(self.clean_prototype_activations))
            contrib_change = self.compute_class_contribution_change(
                self.clean_prototype_activations[:min_len],
                adapted_activations[:min_len],
                labels[:min_len]
            )
            enhanced_metrics.update(contrib_change)
        
        # 4. Adaptation rate (if model tracks it)
        if track_adaptation_rate and hasattr(adapted_model, 'adaptation_stats'):
            stats = adapted_model.adaptation_stats
            enhanced_metrics['adaptation_rate'] = stats.get('adapted_samples', 0) / stats.get('total_samples', 1)
            enhanced_metrics['avg_updates_per_sample'] = stats.get('total_updates', 0) / stats.get('total_samples', 1)
        
        # Merge all metrics
        all_metrics = {**standard_metrics, **enhanced_metrics}
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Enhanced Metrics Summary:")
            print(f"  Accuracy: {all_metrics.get('accuracy', 0)*100:.2f}%")
            print(f"  PAC (Consistency): {all_metrics.get('PAC_mean', 0):.4f}")
            print(f"  PCA (Alignment): {all_metrics.get('PCA_mean', 0):.4f}")
            print(f"  PCA-Weighted (Importance): {all_metrics.get('PCA_weighted_mean', 0):.4f}")
            if 'calibration_agreement' in all_metrics:
                print(f"  Calibration (Agreement): {all_metrics.get('calibration_agreement', 0)*100:.1f}%")
            if 'gt_class_contrib_improvement' in all_metrics:
                print(f"  GT Class Contribution Δ: {all_metrics.get('gt_class_contrib_improvement', 0):+.4f}")
            print(f"{'='*60}")
        
        return all_metrics
