#!/usr/bin/env python3
"""
Prototype-based metrics for ProtoLens Test-Time Adaptation (TTA) quality evaluation.

Adapted from ProtoViT's prototype_tta_metrics.py and enhanced_prototype_metrics.py
for text-based ProtoLens models.

Metrics:
1. PAC (Prototype Activation Consistency) - How stable are activations after adaptation
2. PCA (Prototype Class Alignment) - Are activated prototypes from the correct class
3. Sparsity - Is prototype usage sparse (interpretable)
4. PCA-Weighted - Alignment weighted by FC layer importance
5. Calibration - Prediction agreement with baseline
6. GT Class Contribution - How GT class prototype contributions changed
7. Efficiency - Timing, memory, adapted parameters
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from scipy import stats


class EfficiencyTracker:
    """Track computational efficiency metrics for TTA methods."""
    
    def __init__(self, method_name: str, device: str = 'cuda'):
        self.method_name = method_name
        self.device = device
        
        # Timing
        self.batch_times: List[float] = []
        self.total_samples = 0
        self.start_time = None
        self.end_time = None
        
        # Memory
        self.peak_memory_mb = 0
        
        # Parameters
        self.num_adapted_params = 0
        self.total_params = 0
        
        # Adaptation steps
        self.num_adaptation_steps = 0
        self.total_optimizer_steps = 0
        
    def count_adapted_parameters(self, model, adapted_params: Optional[List] = None):
        """Count how many parameters are being adapted."""
        self.total_params = sum(p.numel() for p in model.parameters())
        
        if adapted_params is not None:
            self.num_adapted_params = sum(p.numel() for p in adapted_params if p.requires_grad)
        else:
            self.num_adapted_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @contextmanager
    def track_inference(self, batch_size: int = 1):
        """Context manager to track inference time for a batch."""
        if self.start_time is None:
            self.start_time = time.time()
        
        # Track memory before
        if 'cuda' in self.device and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        batch_start = time.time()
        
        try:
            yield
        finally:
            if 'cuda' in self.device and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            batch_time = time.time() - batch_start
            self.batch_times.append(batch_time * 1000)  # Convert to ms
            self.total_samples += batch_size
            
            # Track peak memory
            if 'cuda' in self.device and torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
                self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
            
            self.end_time = time.time()
    
    def record_adaptation_step(self, num_steps: int = 1):
        """Record that adaptation steps were performed."""
        self.num_adaptation_steps += num_steps
        self.total_optimizer_steps += num_steps
    
    def get_metrics(self) -> Dict:
        """Get all efficiency metrics as a dictionary."""
        total_time = (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0
        
        return {
            'method_name': self.method_name,
            'total_time_sec': total_time,
            'num_samples': self.total_samples,
            'time_per_sample_ms': (total_time * 1000 / self.total_samples) if self.total_samples > 0 else 0,
            'avg_batch_time_ms': float(np.mean(self.batch_times)) if self.batch_times else 0,
            'std_batch_time_ms': float(np.std(self.batch_times)) if self.batch_times else 0,
            'throughput_samples_per_sec': self.total_samples / total_time if total_time > 0 else 0,
            'num_adapted_params': self.num_adapted_params,
            'total_params': self.total_params,
            'adaptation_ratio': self.num_adapted_params / self.total_params if self.total_params > 0 else 0,
            'num_adaptation_steps': self.num_adaptation_steps,
            'total_optimizer_steps': self.total_optimizer_steps,
            'steps_per_sample': self.num_adaptation_steps / self.total_samples if self.total_samples > 0 else 0,
            'peak_memory_mb': self.peak_memory_mb
        }


class ProtoLensMetricsEvaluator:
    """
    Evaluator for prototype-based TTA metrics in ProtoLens.
    
    Computes:
    - PAC (Prototype Activation Consistency)
    - PCA (Prototype Class Alignment)
    - Sparsity metrics
    - Enhanced metrics (PCA-Weighted, Calibration, GT Class Contribution)
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: ProtoLens model
            device: Device to run on
        """
        self.device = torch.device(device)
        self.num_prototypes = None
        self.num_classes = None
        self.prototype_class_weights = None  # FC layer weights [num_classes, num_prototypes]
        
        # Extract model info
        self._extract_model_info(model)
        
        # Clean baseline data (all stored on self.device)
        self.clean_activations = None
        self.clean_predictions = None
        self.clean_logits = None
        self.clean_labels = None
        
    def _extract_model_info(self, model):
        """Extract prototype and class information from model."""
        # Get the actual model if wrapped
        actual_model = model.model if hasattr(model, 'model') else model
        
        if hasattr(actual_model, 'fc') and hasattr(actual_model.fc, 'weight'):
            self.prototype_class_weights = actual_model.fc.weight.detach().clone().to(self.device)
            self.num_classes, self.num_prototypes = self.prototype_class_weights.shape
        elif hasattr(actual_model, 'proto_linear') and hasattr(actual_model.proto_linear, 'weight'):
            self.prototype_class_weights = actual_model.proto_linear.weight.detach().clone().to(self.device)
            self.num_classes, self.num_prototypes = self.prototype_class_weights.shape
        else:
            print("[Warning] Could not find FC layer weights for prototype-class mapping")
            
    def _forward_no_adapt(self, model: nn.Module, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass without triggering adaptation."""
        actual_model = model.model if hasattr(model, 'model') else model
        
        with torch.no_grad():
            result = actual_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                special_tokens_mask=batch['special_tokens_mask'],
                mode="test",
                original_text=batch.get('original_text', None),
                current_batch_num=0
            )
            
            if isinstance(result, tuple) and len(result) >= 4:
                logits = result[0]
                similarities = result[3]  # [Batch, num_prototypes]
            else:
                logits = result[0] if isinstance(result, tuple) else result
                similarities = None
                
        return logits, similarities
    
    def collect_baseline(self, model: nn.Module, dataloader, max_samples: Optional[int] = None,
                         verbose: bool = True):
        """
        Collect baseline prototype activations for comparison.
        
        Args:
            model: Base model (unadapted)
            dataloader: DataLoader 
            max_samples: Maximum samples to collect
            verbose: Show progress
        """
        if verbose:
            print("Collecting baseline prototype activations...")
        
        all_activations = []
        all_logits = []
        all_predictions = []
        all_labels = []
        n_samples = 0
        
        actual_model = model.model if hasattr(model, 'model') else model
        actual_model.eval()
        
        for batch in dataloader:
            batch_device = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'special_tokens_mask': batch['special_tokens_mask'].to(self.device),
                'original_text': batch.get('original_text', None)
            }
            labels = batch['label'].to(self.device)
            
            logits, similarities = self._forward_no_adapt(model, batch_device)
            
            if similarities is not None:
                all_activations.append(similarities.detach())  # Keep on self.device
            all_logits.append(logits.detach())  # Keep on self.device
            all_predictions.append(logits.argmax(dim=1).detach())  # Keep on self.device
            all_labels.append(labels.detach())  # Keep on self.device
            
            n_samples += labels.size(0)
            if max_samples and n_samples >= max_samples:
                break
        
        # Store all on self.device (CUDA)
        if all_activations:
            self.clean_activations = torch.cat(all_activations, dim=0)
        self.clean_logits = torch.cat(all_logits, dim=0)
        self.clean_predictions = torch.cat(all_predictions, dim=0)
        self.clean_labels = torch.cat(all_labels, dim=0)
        
        if verbose:
            print(f"  Collected baseline from {n_samples} samples")
            if self.clean_activations is not None:
                print(f"  Activation shape: {self.clean_activations.shape}")
    
    def extract_activations(self, model: nn.Module, dataloader, 
                           max_samples: Optional[int] = None,
                           efficiency_tracker: Optional[EfficiencyTracker] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract prototype activations, logits, predictions, and labels from model.
        
        Returns:
            Tuple of (activations, logits, predictions, labels)
        """
        all_activations = []
        all_logits = []
        all_predictions = []
        all_labels = []
        n_samples = 0
        
        for batch in dataloader:
            batch_size = batch['label'].size(0)
            
            batch_device = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'special_tokens_mask': batch['special_tokens_mask'].to(self.device),
                'original_text': batch.get('original_text', None)
            }
            labels = batch['label'].to(self.device)
            
            # Track efficiency if enabled
            if efficiency_tracker:
                with efficiency_tracker.track_inference(batch_size):
                    # Forward through TTA model (may adapt)
                    result = model(
                        input_ids=batch_device['input_ids'],
                        attention_mask=batch_device['attention_mask'],
                        special_tokens_mask=batch_device['special_tokens_mask'],
                        mode="test",
                        original_text=batch_device['original_text'],
                        current_batch_num=0
                    )
            else:
                result = model(
                    input_ids=batch_device['input_ids'],
                    attention_mask=batch_device['attention_mask'],
                    special_tokens_mask=batch_device['special_tokens_mask'],
                    mode="test",
                    original_text=batch_device['original_text'],
                    current_batch_num=0
                )
            
            # Parse result
            if isinstance(result, tuple) and len(result) >= 4:
                logits = result[0]
                similarities = result[3]
            else:
                logits = result[0] if isinstance(result, tuple) else result
                similarities = None
            
            if similarities is not None:
                all_activations.append(similarities.detach())  # Keep on self.device
            all_logits.append(logits.detach())  # Keep on self.device
            all_predictions.append(logits.argmax(dim=1).detach())  # Keep on self.device
            all_labels.append(labels.detach())  # Keep on self.device
            
            n_samples += batch_size
            if max_samples and n_samples >= max_samples:
                break
        
        # Keep all on self.device (CUDA)
        activations = torch.cat(all_activations, dim=0) if all_activations else None
        logits = torch.cat(all_logits, dim=0)
        predictions = torch.cat(all_predictions, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return activations, logits, predictions, labels
    
    def compute_pac(self, adapted_activations: torch.Tensor,
                    clean_activations: Optional[torch.Tensor] = None) -> Dict:
        """
        Compute Prototype Activation Consistency (PAC).
        
        Measures how similar prototype activations are between clean and adapted.
        
        Returns:
            Dict with PAC_mean, PAC_std
        """
        if clean_activations is None:
            clean_activations = self.clean_activations
            
        if clean_activations is None or adapted_activations is None:
            return {'PAC_mean': None, 'PAC_std': None}
        
        # Ensure same device and length
        adapted_activations = adapted_activations.to(self.device)
        clean_activations = clean_activations.to(self.device)
        
        min_len = min(len(clean_activations), len(adapted_activations))
        clean = clean_activations[:min_len]
        adapted = adapted_activations[:min_len]
        
        # Compute cosine similarity per sample
        clean_norm = F.normalize(clean, dim=1)
        adapted_norm = F.normalize(adapted, dim=1)
        
        # Per-sample cosine similarity
        similarities = (clean_norm * adapted_norm).sum(dim=1)
        
        return {
            'PAC_mean': float(similarities.mean().item()),
            'PAC_std': float(similarities.std().item())
        }
    
    def compute_pca(self, activations: torch.Tensor, labels: torch.Tensor,
                    top_k: int = 10) -> Dict:
        """
        Compute Prototype Class Alignment (PCA).
        
        Measures whether activated prototypes belong to the correct class.
        Uses FC layer weights to determine prototype-class relationships.
        
        Returns:
            Dict with PCA_mean, PCA_std
        """
        if activations is None or self.prototype_class_weights is None:
            return {'PCA_mean': None, 'PCA_std': None}
        
        # Ensure same device
        activations = activations.to(self.device)
        labels = labels.to(self.device)
        
        # Get top-K activated prototypes per sample
        _, top_k_indices = torch.topk(activations, min(top_k, activations.size(1)), dim=1)
        
        pca_scores = []
        
        for i, label in enumerate(labels):
            label_idx = label.item()
            sample_top_k = top_k_indices[i]
            
            # Get FC weights for the true class
            class_weights = self.prototype_class_weights[label_idx]  # [num_prototypes]
            
            # Alignment: what fraction of top-K prototypes have positive weight for GT class?
            top_k_weights = class_weights[sample_top_k]
            alignment = (top_k_weights > 0).float().mean().item()
            pca_scores.append(alignment)
        
        return {
            'PCA_mean': float(np.mean(pca_scores)),
            'PCA_std': float(np.std(pca_scores))
        }
    
    def compute_pca_weighted(self, activations: torch.Tensor, labels: torch.Tensor,
                             top_k: int = 10) -> Dict:
        """
        Compute PCA weighted by activation strength AND class importance.
        
        Returns:
            Dict with PCA_weighted_mean, PCA_weighted_std
        """
        if activations is None or self.prototype_class_weights is None:
            return {'PCA_weighted_mean': None, 'PCA_weighted_std': None}
        
        # Ensure same device
        activations = activations.to(self.device)
        labels = labels.to(self.device)
        
        weighted_scores = []
        
        for i, label in enumerate(labels):
            label_idx = label.item()
            sample_acts = activations[i]  # [num_prototypes]
            
            # Get top-K
            top_k_vals, top_k_indices = torch.topk(sample_acts, min(top_k, sample_acts.size(0)))
            
            # Get FC weights for true class
            class_weights = self.prototype_class_weights[label_idx]
            top_k_class_weights = class_weights[top_k_indices]
            
            # Weighted alignment: activation * class_weight (positive = good)
            # Normalize activations to [0, 1]
            act_normalized = F.softmax(top_k_vals, dim=0)
            
            # Compute weighted contribution
            contributions = act_normalized * top_k_class_weights
            positive_contribution = contributions[contributions > 0].sum().item()
            total_contribution = contributions.abs().sum().item()
            
            score = positive_contribution / (total_contribution + 1e-8)
            weighted_scores.append(score)
        
        return {
            'PCA_weighted_mean': float(np.mean(weighted_scores)),
            'PCA_weighted_std': float(np.std(weighted_scores))
        }
    
    def compute_sparsity(self, activations: torch.Tensor, threshold: float = 0.1) -> Dict:
        """
        Compute sparsity of prototype activations.
        
        Returns:
            Dict with sparsity_gini_mean, sparsity_active_mean
        """
        if activations is None:
            return {'sparsity_gini_mean': None, 'sparsity_active_mean': None}
        
        def gini(array):
            """Calculate Gini coefficient (0=equality, 1=inequality)."""
            array = np.array(array).flatten()
            array = array + 1e-8  # Avoid zeros
            array = np.sort(array)
            n = len(array)
            index = np.arange(1, n + 1)
            return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)
        
        # Move to CPU for numpy operations
        activations_cpu = activations.cpu()
        
        gini_scores = []
        active_counts = []
        
        for i in range(len(activations_cpu)):
            sample_acts = activations_cpu[i].numpy()
            
            # Normalize to positive for Gini
            sample_acts_pos = sample_acts - sample_acts.min() + 1e-8
            
            gini_scores.append(gini(sample_acts_pos))
            active_counts.append((sample_acts > threshold).sum())
        
        return {
            'sparsity_gini_mean': float(np.mean(gini_scores)),
            'sparsity_active_mean': float(np.mean(active_counts))
        }
    
    def compute_calibration(self, adapted_predictions: torch.Tensor,
                           adapted_logits: torch.Tensor) -> Dict:
        """
        Compute calibration metrics comparing to baseline.
        
        Returns:
            Dict with calibration_agreement, calibration_logit_corr
        """
        if self.clean_predictions is None or self.clean_logits is None:
            return {'calibration_agreement': None, 'calibration_logit_corr': None}
        
        # Move to CPU for comparison (needed for numpy/scipy)
        adapted_predictions = adapted_predictions.cpu()
        adapted_logits = adapted_logits.cpu()
        clean_predictions = self.clean_predictions.cpu()
        clean_logits = self.clean_logits.cpu()
        
        # Ensure same length
        min_len = min(len(clean_predictions), len(adapted_predictions))
        clean_preds = clean_predictions[:min_len]
        adapted_preds = adapted_predictions[:min_len]
        clean_logs = clean_logits[:min_len]
        adapted_logs = adapted_logits[:min_len]
        
        # Prediction agreement
        agreement = (clean_preds == adapted_preds).float().mean().item()
        
        # Logit correlation (average per sample)
        correlations = []
        for i in range(min_len):
            clean_l = clean_logs[i].numpy()
            adapted_l = adapted_logs[i].numpy()
            corr, _ = stats.pearsonr(clean_l, adapted_l)
            if not np.isnan(corr):
                correlations.append(corr)
        
        avg_corr = float(np.mean(correlations)) if correlations else 0.0
        
        return {
            'calibration_agreement': agreement,
            'calibration_logit_corr': avg_corr
        }
    
    def compute_gt_class_contribution(self, adapted_activations: torch.Tensor,
                                       labels: torch.Tensor) -> Dict:
        """
        Compute how GT class prototype contributions changed.
        
        Returns:
            Dict with gt_class_contrib_improvement, gt_class_contrib_change_mean
        """
        if self.clean_activations is None or adapted_activations is None:
            return {'gt_class_contrib_improvement': None, 'gt_class_contrib_change_mean': None}
        
        if self.prototype_class_weights is None:
            return {'gt_class_contrib_improvement': None, 'gt_class_contrib_change_mean': None}
        
        # Ensure same device
        adapted_activations = adapted_activations.to(self.device)
        clean_activations = self.clean_activations.to(self.device)
        labels = labels.to(self.device)
        
        min_len = min(len(clean_activations), len(adapted_activations))
        clean_acts = clean_activations[:min_len]
        adapted_acts = adapted_activations[:min_len]
        labels = labels[:min_len]
        
        contribution_changes = []
        
        for i, label in enumerate(labels):
            label_idx = label.item()
            class_weights = self.prototype_class_weights[label_idx]  # Already on self.device
            
            # Compute weighted contribution for GT class
            clean_contrib = (clean_acts[i] * class_weights).sum().item()
            adapted_contrib = (adapted_acts[i] * class_weights).sum().item()
            
            change = adapted_contrib - clean_contrib
            contribution_changes.append(change)
        
        return {
            'gt_class_contrib_improvement': float(np.sum(contribution_changes)),
            'gt_class_contrib_change_mean': float(np.mean(contribution_changes))
        }
    
    def evaluate(self, model: nn.Module, dataloader, 
                 top_k: int = 10,
                 max_samples: Optional[int] = None,
                 efficiency_tracker: Optional[EfficiencyTracker] = None,
                 verbose: bool = True) -> Dict:
        """
        Full evaluation of a TTA method using all prototype-based metrics.
        
        Args:
            model: TTA-adapted model
            dataloader: Test data loader
            top_k: Number of top prototypes for alignment metrics
            max_samples: Max samples to process
            efficiency_tracker: Optional tracker for efficiency metrics
            verbose: Show progress
            
        Returns:
            Dict with all metrics
        """
        if verbose:
            print("Computing prototype-based metrics...")
        
        # Extract activations from adapted model
        activations, logits, predictions, labels = self.extract_activations(
            model, dataloader, max_samples, efficiency_tracker
        )
        
        # Compute accuracy
        accuracy = (predictions == labels).float().mean().item()
        
        # Compute all metrics
        results = {'accuracy': accuracy}
        
        # PAC
        pac_metrics = self.compute_pac(activations)
        results.update(pac_metrics)
        
        # PCA
        pca_metrics = self.compute_pca(activations, labels, top_k)
        results.update(pca_metrics)
        
        # Sparsity
        sparsity_metrics = self.compute_sparsity(activations)
        results.update(sparsity_metrics)
        
        # PCA Weighted
        pca_weighted_metrics = self.compute_pca_weighted(activations, labels, top_k)
        results.update(pca_weighted_metrics)
        
        # Calibration
        calibration_metrics = self.compute_calibration(predictions, logits)
        results.update(calibration_metrics)
        
        # GT Class Contribution
        gt_contrib_metrics = self.compute_gt_class_contribution(activations, labels)
        results.update(gt_contrib_metrics)
        
        # Efficiency metrics
        if efficiency_tracker:
            results['efficiency'] = efficiency_tracker.get_metrics()
        
        # Adaptation stats from model
        if hasattr(model, 'adaptation_stats'):
            stats = model.adaptation_stats
            results['adaptation_rate'] = stats.get('adapted_samples', 0) / max(stats.get('total_samples', 1), 1)
            results['avg_updates_per_sample'] = stats.get('total_updates', 0) / max(stats.get('total_samples', 1), 1)
            results['adaptation_stats'] = {
                'total_samples': stats.get('total_samples', 0),
                'adapted_samples': stats.get('adapted_samples', 0),
                'total_updates': stats.get('total_updates', stats.get('adapted_samples', 0))
            }
        
        if verbose:
            print(f"  Accuracy: {accuracy*100:.2f}%")
            if results.get('PAC_mean') is not None:
                print(f"  PAC: {results['PAC_mean']*100:.2f}%")
            if results.get('PCA_mean') is not None:
                print(f"  PCA: {results['PCA_mean']*100:.2f}%")
            if results.get('sparsity_gini_mean') is not None:
                print(f"  Sparsity (Gini): {results['sparsity_gini_mean']:.3f}")
        
        return results
