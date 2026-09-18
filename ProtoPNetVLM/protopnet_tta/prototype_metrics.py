#!/usr/bin/env python3
"""
Prototype-based metrics for Test-Time Adaptation (TTA) quality evaluation.
Adapted from ProtoViT for ProtoPNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class PrototypeMetricsEvaluator:
    """Evaluator for prototype-based TTA metrics."""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: ProtoPNet model (used to extract prototype info)
            device: Device to run on
        """
        self.device = device

        # Store the full model for forward passes
        self.ppnet = model

        # Extract core for prototype information
        if hasattr(model, 'core'):
            core = model.core
        elif hasattr(model, 'model'):
            # Wrapped model
            if hasattr(model.model, 'core'):
                core = model.model.core
            else:
                core = model.model
        else:
            core = model

        # Get prototype class identity: [num_prototypes, num_classes]
        self.prototype_class_identity = core.prototype_class_identity.to(device)
        self.proto_identities = self.prototype_class_identity.argmax(dim=1)  # [num_prototypes]

        # Get number of prototypes and classes
        self.num_prototypes = self.prototype_class_identity.shape[0]
        self.num_classes = self.prototype_class_identity.shape[1]

        # Storage for clean baseline
        self.clean_prototype_activations = None  # Will store clean activations
        self.clean_labels = None

    def _forward_no_adapt(self, model: nn.Module, images: torch.Tensor):
        """Forward pass without triggering additional adaptation steps."""
        if hasattr(model, 'model'):
            return model.model(images)
        return model(images)

    def extract_prototype_activations(self, model: nn.Module, loader,
                                     max_samples: Optional[int] = None,
                                     verbose: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract prototype activations from all samples in loader.

        Returns:
            activations: [N, num_prototypes] - prototype activations for each sample
            labels: [N] - ground truth labels
        """
        all_activations = []
        all_labels = []

        sample_count = 0

        with torch.no_grad():
            iterator = tqdm(loader, desc="Extracting prototypes") if verbose else loader

            for images, labels in iterator:
                if max_samples and sample_count >= max_samples:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass without adaptation
                outputs = self._forward_no_adapt(model, images)

                # Extract prototype similarities
                # ProtoPNet returns tuple: (logits, min_distances)
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    min_distances = outputs[1]  # [B, P]
                    # Convert distances to similarities
                    # Using inverse distance: similarity = 1 / (1 + distance)
                    prototype_activations = torch.log((min_distances + 1.0) / (min_distances + 1e-4))
                else:
                    raise ValueError("Model output format not supported. Expected (logits, min_distances)")

                all_activations.append(prototype_activations.cpu())
                all_labels.append(labels.cpu())

                sample_count += images.size(0)

        # Concatenate all batches
        activations = torch.cat(all_activations, dim=0)  # [N, P]
        labels = torch.cat(all_labels, dim=0)  # [N]

        if max_samples:
            activations = activations[:max_samples]
            labels = labels[:max_samples]

        return activations, labels

    def collect_clean_baseline(self, clean_loader, max_samples: Optional[int] = None,
                               verbose: bool = True):
        """
        Collect clean prototype activations as baseline for comparison.

        Args:
            clean_loader: DataLoader with clean (uncorrupted) images
            max_samples: Maximum samples to collect (None = all)
            verbose: Show progress
        """
        print("Collecting clean baseline prototype activations...")

        self.clean_prototype_activations, self.clean_labels = self.extract_prototype_activations(
            self.ppnet, clean_loader, max_samples=max_samples, verbose=verbose
        )

        print(f"✓ Collected clean baseline: {self.clean_prototype_activations.shape[0]} samples")

    def compute_prototype_activation_consistency(
        self,
        adapted_activations: torch.Tensor,
        clean_activations: Optional[torch.Tensor] = None,
        method: str = 'cosine'
    ) -> Dict[str, float]:
        """
        Compute Prototype Activation Consistency (PAC).
        Measures how similar prototype activations are between clean and adapted images.

        Args:
            adapted_activations: [N, P] - activations after adaptation
            clean_activations: [N, P] - activations on clean images (None = use stored baseline)
            method: 'cosine', 'l2', or 'correlation'

        Returns:
            Dictionary with consistency metrics
        """
        if clean_activations is None:
            if self.clean_prototype_activations is None:
                raise ValueError("No clean baseline collected. Call collect_clean_baseline() first.")
            clean_activations = self.clean_prototype_activations

        # Paired metrics must never silently truncate mismatched cohorts.
        if len(adapted_activations) != len(clean_activations):
            raise ValueError(
                'PAC pairing mismatch: '
                f'adapted={len(adapted_activations)}, '
                f'clean={len(clean_activations)}'
            )
        n_samples = len(adapted_activations)

        if method == 'cosine':
            # Cosine similarity per sample
            similarities = F.cosine_similarity(
                adapted_activations.float(),
                clean_activations.float(),
                dim=1
            )
            consistency_per_sample = similarities.numpy()

        elif method == 'l2':
            # Normalized L2 distance (convert to similarity)
            l2_distances = torch.norm(adapted_activations - clean_activations, dim=1, p=2)
            # Normalize by vector magnitude
            norm_clean = torch.norm(clean_activations, dim=1, p=2)
            normalized_distances = l2_distances / (norm_clean + 1e-8)
            # Convert distance to similarity (1 - normalized_distance)
            consistency_per_sample = (1 - normalized_distances).clamp(0, 1).numpy()

        elif method == 'correlation':
            # Pearson correlation per sample
            consistency_per_sample = []
            for i in range(n_samples):
                corr = np.corrcoef(
                    adapted_activations[i].numpy(),
                    clean_activations[i].numpy()
                )[0, 1]
                consistency_per_sample.append(corr if not np.isnan(corr) else 0.0)
            consistency_per_sample = np.array(consistency_per_sample)

        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'PAC_mean': float(np.mean(consistency_per_sample)),
            'PAC_std': float(np.std(consistency_per_sample)),
            'PAC_median': float(np.median(consistency_per_sample)),
            'PAC_min': float(np.min(consistency_per_sample)),
            'PAC_max': float(np.max(consistency_per_sample)),
        }

    def compute_prototype_class_alignment(
        self,
        prototype_activations: torch.Tensor,
        labels: torch.Tensor,
        top_k: int = 10,
        weight_by_activation: bool = True
    ) -> Dict[str, float]:
        """
        Compute Prototype Class Alignment (PCA).
        Measures whether activated prototypes belong to the correct class.

        Args:
            prototype_activations: [N, P] - prototype activations
            labels: [N] - ground truth class labels
            top_k: Number of top prototypes to consider
            weight_by_activation: Weight alignment by activation strength

        Returns:
            Dictionary with alignment metrics
        """
        n_samples = prototype_activations.shape[0]
        alignment_scores = []

        # Move proto_identities to CPU to match prototype_activations device
        proto_identities_cpu = self.proto_identities.cpu()

        for i in range(n_samples):
            activations = prototype_activations[i]  # [P]
            true_label = labels[i].item()

            # Get top-k activated prototypes
            top_k_values, top_k_indices = torch.topk(activations, k=min(top_k, len(activations)))

            # Check which top-k prototypes belong to the true class
            # Use CPU version to match device
            top_k_proto_classes = proto_identities_cpu[top_k_indices]  # [k]
            correct_class_mask = (top_k_proto_classes == true_label).float()  # [k]

            if weight_by_activation:
                # Weight by activation strength (softmax for normalization)
                weights = F.softmax(top_k_values.float(), dim=0)
                alignment = (correct_class_mask * weights).sum().item()
            else:
                # Simple proportion of correct-class prototypes
                alignment = correct_class_mask.mean().item()

            alignment_scores.append(alignment)

        alignment_scores = np.array(alignment_scores)

        return {
            'PCA_mean': float(np.mean(alignment_scores)),
            'PCA_std': float(np.std(alignment_scores)),
            'PCA_median': float(np.median(alignment_scores)),
            'PCA_min': float(np.min(alignment_scores)),
            'PCA_max': float(np.max(alignment_scores)),
        }

    def compute_prototype_activation_sparsity(
        self,
        prototype_activations: torch.Tensor,
        threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Compute sparsity of prototype activations.
        High sparsity = few prototypes activate strongly (good for interpretability).

        Args:
            prototype_activations: [N, P]
            threshold: Activation threshold for counting as "active"

        Returns:
            Dictionary with sparsity metrics
        """
        # Gini coefficient for sparsity
        def gini(array):
            """
            Calculate Gini coefficient (0=perfect equality, 1=perfect inequality).
            Higher Gini = more sparse (few prototypes dominate).
            """
            array = np.abs(array)
            if array.sum() == 0:
                return 0.0
            # Sort in ascending order
            sorted_array = np.sort(array)
            n = len(array)
            cumsum = np.cumsum(sorted_array)
            return (n + 1 - 2 * np.sum(cumsum) / array.sum()) / n

        gini_scores = []
        active_proto_counts = []

        for i in range(len(prototype_activations)):
            activations = prototype_activations[i].numpy()

            # Gini coefficient
            gini_score = gini(activations)
            gini_scores.append(gini_score)

            # Count of active prototypes (above threshold)
            active_count = (np.abs(activations) > threshold).sum()
            active_proto_counts.append(active_count)

        gini_scores = np.array(gini_scores)
        active_proto_counts = np.array(active_proto_counts)

        return {
            'sparsity_gini_mean': float(np.mean(gini_scores)),
            'sparsity_gini_std': float(np.std(gini_scores)),
            'sparsity_active_mean': float(np.mean(active_proto_counts)),
            'sparsity_active_std': float(np.std(active_proto_counts)),
        }

    def evaluate_tta_method(
        self,
        adapted_model: nn.Module,
        test_loader,
        top_k: int = 10,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Full evaluation of a TTA method using prototype-based metrics.

        Args:
            adapted_model: TTA-adapted model
            test_loader: Test data loader (corrupted images)
            top_k: Number of top prototypes for PCA
            max_samples: Max samples to evaluate (None = all)
            verbose: Show progress

        Returns:
            Dictionary with all metrics
        """
        if verbose:
            print("Evaluating TTA method with prototype-based metrics...")

        # Extract prototype activations from adapted model
        adapted_activations, labels = self.extract_prototype_activations(
            adapted_model, test_loader, max_samples=max_samples, verbose=verbose
        )

        metrics = {}

        # 1. Prototype Activation Consistency (PAC)
        if self.clean_prototype_activations is not None:
            if verbose:
                print("Computing Prototype Activation Consistency (PAC)...")
            pac_metrics = self.compute_prototype_activation_consistency(
                adapted_activations,
                method='cosine'
            )
            metrics.update(pac_metrics)
        else:
            if verbose:
                print("⚠ Skipping PAC (no clean baseline)")

        # 2. Prototype Class Alignment (PCA)
        if verbose:
            print("Computing Prototype Class Alignment (PCA)...")
        pca_metrics = self.compute_prototype_class_alignment(
            adapted_activations,
            labels,
            top_k=top_k,
            weight_by_activation=True
        )
        metrics.update(pca_metrics)

        # 3. Prototype Activation Sparsity
        if verbose:
            print("Computing Prototype Activation Sparsity...")
        sparsity_metrics = self.compute_prototype_activation_sparsity(
            adapted_activations,
            threshold=0.1
        )
        metrics.update(sparsity_metrics)

        if verbose:
            print(f"\n{'='*60}")
            print(f"PAC (Consistency): {metrics.get('PAC_mean', 0)*100:.2f}%")
            print(f"PCA (Class Alignment): {metrics.get('PCA_mean', 0)*100:.2f}%")
            print(f"Sparsity (Gini): {metrics.get('sparsity_gini_mean', 0):.3f}")
            print(f"{'='*60}")

        return metrics
