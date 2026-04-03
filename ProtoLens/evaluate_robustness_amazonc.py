"""
Comprehensive Robustness Evaluation for ProtoLens on Amazon-C.

Features:
- Incremental saving with backup (resumable if interrupted)
- Additive mode (only adds new methods/corruptions, doesn't overwrite)
- Override mode (--override <methods> to re-run specific methods)
- Supports: baseline, tent, eata, prototta, sar
- Includes prototype-based metrics (PAC, PCA, etc.)
- Matches ProtoViT evaluation format

Usage:
    # Run all methods
    CUDA_VISIBLE_DEVICES=0 python evaluate_robustness_amazonc.py
    
    # Run specific method (additive)
    CUDA_VISIBLE_DEVICES=0 python evaluate_robustness_amazonc.py --methods prototta
    
    # Override specific method (re-run and replace in JSON)
    CUDA_VISIBLE_DEVICES=0 python evaluate_robustness_amazonc.py --methods prototta --override prototta
    
    # Quick mode
    CUDA_VISIBLE_DEVICES=0 python evaluate_robustness_amazonc.py --quick

Output:
    Datasets/Amazon-C/results/robustness_results.json (updated incrementally)
"""

import os
import sys
import json
import shutil
import argparse
import math
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import AutoTokenizer

# Import ProtoLens modules
from PLens import BERTClassifier
from utils import TextClassificationDataset
import tent
import eata
import proto_tta
import adapt_utils
from prototype_metrics import ProtoLensMetricsEvaluator, EfficiencyTracker

# Set seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)


# ============================================================================
# Configuration
# ============================================================================

class Cfg:
    """Configuration for TTA optimization."""
    def __init__(self):
        self.OPTIM = self.Optim()
        self.MODEL = self.Model()

    class Optim:
        def __init__(self):
            self.METHOD = 'Adam'
            self.LR = 0.000005  # Default LR
            self.BETA = 0.9
            self.WD = 0.0
            self.STEPS = 1

    class Model:
        def __init__(self):
            self.EPISODIC = False

cfg = Cfg()


# Default corruptions and severities
ALL_CORRUPTION_TYPES = ['qwerty', 'swap', 'remove_char', 'mixed', 'aggressive']
ALL_SEVERITIES = [20, 40, 60, 80]

QUICK_CORRUPTION_TYPES = ['qwerty', 'aggressive']
QUICK_SEVERITIES = [40, 60]


# ============================================================================
# SAR Implementation for Text (adapted from ProtoViT)
# ============================================================================

class SAM(optim.Optimizer):
    """Sharpness-Aware Minimization optimizer."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    with torch.no_grad():
        return 0.9 * ema + (1 - 0.9) * new_data


class SAR(nn.Module):
    """SAR for text classification with ProtoLens."""
    
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 margin_e0=0.4*math.log(2), reset_constant_em=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.margin_e0 = margin_e0
        self.reset_constant_em = reset_constant_em
        self.ema = None
        
        # Save initial state
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())
        
        self.adaptation_stats = {
            'total_samples': 0,
            'adapted_samples': 0,
            'total_updates': 0,
        }

    def forward(self, input_ids=None, attention_mask=None, special_tokens_mask=None,
                mode="test", original_text=None, current_batch_num=None, **kwargs):
        if self.episodic:
            self.reset()
        
        batch_size = input_ids.size(0) if input_ids is not None else 1
        self.adaptation_stats['total_samples'] += batch_size
        
        for _ in range(self.steps):
            result, ema, reset_flag, num_adapted = self._forward_and_adapt_sar(
                input_ids, attention_mask, special_tokens_mask, 
                mode, original_text, current_batch_num
            )
            if reset_flag:
                self.reset()
            self.ema = ema
            self.adaptation_stats['adapted_samples'] += num_adapted
            if num_adapted > 0:
                self.adaptation_stats['total_updates'] += num_adapted
        
        # Return full tuple: (outputs, loss_mu, augmented_loss, similarity)
        if isinstance(result, tuple) and len(result) >= 4:
            return result[0], result[1], result[2], result[3]
        elif isinstance(result, tuple):
            return result[0], result[1] if len(result) > 1 else None, result[2] if len(result) > 2 else None, None
        return result, None, None, None

    def _forward_and_adapt_sar(self, input_ids, attention_mask, special_tokens_mask,
                                mode, original_text, current_batch_num):
        self.optimizer.zero_grad()
        
        # Forward - ProtoLens returns (logits, loss_mu, augmented_loss, similarity)
        result = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            mode=mode,
            original_text=original_text,
            current_batch_num=current_batch_num
        )
        outputs = result[0] if isinstance(result, tuple) else result
        
        # Entropy filtering
        entropys = -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
        filter_ids_1 = torch.where(entropys < self.margin_e0)
        num_filtered_1 = len(filter_ids_1[0])
        
        if num_filtered_1 == 0:
            return result, self.ema, False, 0
        
        entropys_filtered = entropys[filter_ids_1]
        loss = entropys_filtered.mean()
        loss.backward()
        
        self.optimizer.first_step(zero_grad=True)
        
        # Second forward at perturbed weights
        result2 = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            mode=mode,
            original_text=original_text,
            current_batch_num=current_batch_num
        )
        outputs2 = result2[0] if isinstance(result2, tuple) else result2
        
        entropys2 = -(outputs2.softmax(1) * outputs2.log_softmax(1)).sum(1)
        entropys2 = entropys2[filter_ids_1]
        loss_second_value = entropys2.clone().detach().mean()
        
        filter_ids_2 = torch.where(entropys2 < self.margin_e0)
        num_filtered_2 = len(filter_ids_2[0])
        
        if num_filtered_2 == 0:
            return result, self.ema, False, 0
        
        loss_second = entropys2[filter_ids_2].mean()
        if not np.isnan(loss_second.item()):
            self.ema = update_ema(self.ema, loss_second.item())
        
        loss_second.backward()
        self.optimizer.second_step(zero_grad=True)
        
        reset_flag = self.ema is not None and self.ema < 0.2
        
        # Return the result from the second forward (more up-to-date)
        return result2, self.ema, reset_flag, num_filtered_2

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.ema = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


# ============================================================================
# Utility Functions
# ============================================================================

def setup_optimizer(params):
    """Set up optimizer for TTA adaptation."""
    return optim.Adam(params, lr=cfg.OPTIM.LR, betas=(cfg.OPTIM.BETA, 0.999), 
                     weight_decay=cfg.OPTIM.WD)


def load_model(model_path, device):
    """Load trained ProtoLens model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    saved_args = checkpoint.get('pnfrl_args', {})
    
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(
        bert_model_name=saved_args.get('bert_model_name', 'all-mpnet-base-v2'),
        num_classes=saved_args.get('num_classes', 2),
        prototype_num=saved_args.get('prototype_num', 50),
        batch_size=saved_args.get('batch_size', 32),
        hidden_dim=saved_args.get('hidden_dim', 768),
        max_length=saved_args.get('max_length', 512),
        data_set='Yelp', base_folder='Datasets', gaussian_num=6, window_size=5
    )
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    
    model = BERTClassifier(
        args=args, bert_model_name='sentence-transformers/all-mpnet-base-v2',
        num_classes=args.num_classes, num_prototype=args.prototype_num,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim,
        max_length=args.max_length, tokenizer=tokenizer
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, tokenizer, args


def load_corrupted_data(data_dir, corruption_type, severity):
    """Load corrupted Amazon-C dataset."""
    filename = f'amazon_c_{corruption_type}_s{severity}.csv'
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    return pd.read_csv(filepath)


def load_clean_data(data_dir):
    """Load clean (uncorrupted) Amazon test dataset for baseline reference.
    
    The clean data is typically in the parent 'Amazon' directory, 
    while corrupted data is in 'Amazon-C'.
    """
    # First, check if data_dir points to Amazon-C, and look in parent Amazon folder
    parent_dir = os.path.dirname(data_dir.rstrip('/'))
    amazon_dir = os.path.join(parent_dir, 'Amazon')
    
    # Priority 1: Look in parent Amazon directory for test.csv
    if os.path.exists(amazon_dir):
        test_path = os.path.join(amazon_dir, 'test.csv')
        if os.path.exists(test_path):
            return pd.read_csv(test_path)
    
    # Priority 2: Try common clean data filenames in current data_dir
    for clean_name in ['amazon_test.csv', 'amazon_clean.csv', 'test.csv']:
        filepath = os.path.join(data_dir, clean_name)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
    
    # Priority 3: Fallback - use severity 0 if available (no corruption)
    for corruption in ['qwerty', 'swap', 'remove_char', 'mixed', 'aggressive']:
        filepath = os.path.join(data_dir, f'amazon_c_{corruption}_s0.csv')
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
    
    raise FileNotFoundError(f"No clean data found. Looked in {amazon_dir} and {data_dir}")


def create_dataloader(df, tokenizer, max_length, batch_size):
    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                     pin_memory=True, num_workers=4)


# ============================================================================
# TTA Method Setup Functions (matching run_inference_amazon_c.py)
# ============================================================================

def setup_tent(model, adaptation_mode):
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    print(f"TENT: Adapting {len(params)} parameter groups")
    optimizer = setup_optimizer(params)
    return tent.Tent(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def setup_eata(model, adaptation_mode, e_margin, d_margin):
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    print(f"EATA: Adapting {len(params)} parameter groups")
    print(f"  E-margin: {e_margin}, D-margin: {d_margin}")
    optimizer = setup_optimizer(params)
    return eata.EATA(model, optimizer, fishers=None, steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC, e_margin=e_margin, d_margin=d_margin)


def setup_prototta(model, adaptation_mode, use_geo_filter, geo_threshold,
                   importance_mode='global', sigmoid_temperature=5.0):
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    print(f"ProtoTTA: Adapting {len(params)} parameter groups")
    print(f"  Geometric filter: {use_geo_filter}, Threshold: {geo_threshold}")
    print(f"  Sigmoid temperature: {sigmoid_temperature}")
    optimizer = setup_optimizer(params)
    return proto_tta.ProtoTTA(model, optimizer, steps=cfg.OPTIM.STEPS,
                              episodic=cfg.MODEL.EPISODIC,
                              use_geometric_filter=use_geo_filter,
                              geo_filter_threshold=geo_threshold,
                              consensus_strategy='max',
                              importance_mode=importance_mode,
                              sigmoid_temperature=sigmoid_temperature)


def setup_sar(model, adaptation_mode):
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, _ = adapt_utils.collect_params(model, adaptation_mode)
    base_optimizer = optim.SGD
    optimizer = SAM(params, base_optimizer, lr=cfg.OPTIM.LR, momentum=0.9)
    return SAR(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_baseline(model, dataloader, device, 
                       metrics_evaluator: Optional[ProtoLensMetricsEvaluator] = None,
                       track_efficiency: bool = False):
    """Evaluate model without adaptation."""
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    all_logits = []
    all_similarities = []
    
    # Setup efficiency tracker
    efficiency_tracker = None
    if track_efficiency:
        efficiency_tracker = EfficiencyTracker('baseline', device=str(device))
        efficiency_tracker.count_adapted_parameters(model, adapted_params=[])
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Baseline", ncols=100):
            batch_size = batch['label'].size(0)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)
            labels = batch['label'].to(device)
            
            if efficiency_tracker:
                with efficiency_tracker.track_inference(batch_size):
                    result = model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        special_tokens_mask=special_tokens_mask, mode="test",
                        original_text=batch['original_text'], current_batch_num=0
                    )
            else:
                result = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask, mode="test",
                    original_text=batch['original_text'], current_batch_num=0
                )
                
            if isinstance(result, tuple) and len(result) >= 4:
                outputs = result[0]
                similarities = result[3]
                # Ensure similarities has consistent shape across batches
                # If it has 3 dims (batch, seq, prototypes), reduce to (batch, prototypes)
                if similarities.dim() == 3:
                    # Take mean or max across sequence dimension
                    similarities = similarities.mean(dim=1)
                # Ensure it's at least 2D
                if similarities.dim() == 1:
                    similarities = similarities.unsqueeze(-1)
                all_similarities.append(similarities.detach().cpu())  # Store on CPU
            else:
                outputs = result[0] if isinstance(result, tuple) else result
            
            probs = torch.softmax(outputs, dim=1)
            all_logits.append(outputs.detach().cpu())  # Store on CPU
            
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())
    
    # Collect baseline for metrics evaluator (keep on CUDA)
    # Note: This is used when baseline IS the reference (old behavior)
    # For the new clean-reference behavior, we don't overwrite the evaluator's clean data
    
    all_probs = np.array(all_probs)
    
    result_dict = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'avg_confidence': float(np.mean(np.max(all_probs, axis=1))),
        'num_samples': len(all_labels)
    }
    
    # Return raw data for external PAC/PCA computation (comparing to clean reference)
    if all_similarities:
        result_dict['_similarities'] = torch.cat(all_similarities, dim=0)
        result_dict['_logits'] = torch.cat(all_logits, dim=0)
        result_dict['_predictions'] = torch.tensor(all_preds)
        result_dict['_labels'] = torch.tensor(all_labels)
    
    if efficiency_tracker:
        result_dict['efficiency'] = efficiency_tracker.get_metrics()
        
    return result_dict


def evaluate_tta_method(tta_model, dataloader, device, description="TTA", 
                        metrics_evaluator: Optional[ProtoLensMetricsEvaluator] = None,
                        track_efficiency: bool = False,
                        compute_proto_metrics: bool = True):
    """Evaluate with TTA adaptation."""
    all_preds, all_labels = [], []
    all_probs = []
    all_logits = []
    all_similarities = []
    
    # Setup efficiency tracker
    efficiency_tracker = None
    if track_efficiency:
        method_name = description.lower().split()[0]
        efficiency_tracker = EfficiencyTracker(method_name, device=str(device))
        # Count adapted parameters
        if hasattr(tta_model, 'model'):
            adapted_params = [p for p in tta_model.model.parameters() if p.requires_grad]
            efficiency_tracker.count_adapted_parameters(tta_model.model, adapted_params)
        else:
            adapted_params = [p for p in tta_model.parameters() if p.requires_grad]
            efficiency_tracker.count_adapted_parameters(tta_model, adapted_params)
    
    num_batches = 0
    for batch in tqdm(dataloader, desc=description, ncols=100):
        batch_size = batch['label'].size(0)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        labels = batch['label'].to(device)
        
        try:
            if efficiency_tracker:
                with efficiency_tracker.track_inference(batch_size):
                    result = tta_model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        special_tokens_mask=special_tokens_mask, mode="test",
                        original_text=batch['original_text'], current_batch_num=num_batches
                    )
            else:
                result = tta_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    special_tokens_mask=special_tokens_mask, mode="test",
                    original_text=batch['original_text'], current_batch_num=num_batches
                )
                
            if isinstance(result, tuple) and len(result) >= 4:
                outputs = result[0]
                similarities = result[3]
                # Ensure similarities has consistent shape across batches
                # If it has 3 dims (batch, seq, prototypes), reduce to (batch, prototypes)
                if similarities.dim() == 3:
                    # Take mean or max across sequence dimension
                    similarities = similarities.mean(dim=1)
                # Ensure it's at least 2D
                if similarities.dim() == 1:
                    similarities = similarities.unsqueeze(-1)
                all_similarities.append(similarities.detach().cpu())  # Store on CPU
            else:
                outputs = result[0] if isinstance(result, tuple) else result
                
            probs = torch.softmax(outputs, dim=1)
            all_logits.append(outputs.detach().cpu())  # Store on CPU
        except Exception as e:
            print(f"Error in batch: {e}")
            continue
        
        _, preds = torch.max(outputs, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().detach().numpy())
        num_batches += 1
    
    # Record adaptation steps
    if efficiency_tracker:
        efficiency_tracker.record_adaptation_step(num_batches)
        
    all_probs = np.array(all_probs) if all_probs else np.array([])
    
    result_dict = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'avg_confidence': float(np.mean(np.max(all_probs, axis=1))) if len(all_probs) > 0 else 0,
        'num_samples': len(all_labels)
    }
    
    # Compute prototype metrics if evaluator is provided
    if compute_proto_metrics and metrics_evaluator is not None and all_similarities:
        adapted_activations = torch.cat(all_similarities, dim=0)
        adapted_logits = torch.cat(all_logits, dim=0)
        adapted_predictions = torch.tensor(all_preds)
        labels_tensor = torch.tensor(all_labels)
        
        result_dict.update(metrics_evaluator.compute_pac(adapted_activations))
        result_dict.update(metrics_evaluator.compute_pca(adapted_activations, labels_tensor))
        result_dict.update(metrics_evaluator.compute_sparsity(adapted_activations))
        result_dict.update(metrics_evaluator.compute_pca_weighted(adapted_activations, labels_tensor))
        result_dict.update(metrics_evaluator.compute_calibration(adapted_predictions, adapted_logits))
        result_dict.update(metrics_evaluator.compute_gt_class_contribution(adapted_activations, labels_tensor))
    
    # Add adaptation stats
    if hasattr(tta_model, 'adaptation_stats'):
        stats = tta_model.adaptation_stats
        total_samples = stats.get('total_samples', len(all_labels))
        adapted_samples = stats.get('adapted_samples', 0)
        total_updates = stats.get('total_updates', adapted_samples)
        
        result_dict['adaptation_rate'] = adapted_samples / max(total_samples, 1)
        result_dict['avg_updates_per_sample'] = total_updates / max(total_samples, 1)
        result_dict['adaptation_stats'] = {
            'total_samples': total_samples,
            'adapted_samples': adapted_samples,
            'total_updates': total_updates
        }
    
    if efficiency_tracker:
        result_dict['efficiency'] = efficiency_tracker.get_metrics()
        
    return result_dict


# ============================================================================
# Results Management (Incremental Save with Backup)
# ============================================================================

def load_results_json(output_file):
    """Load existing results from JSON file."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load existing results: {e}")
    return None


def save_results_json(output_file, new_data):
    """Save results to JSON with backup and merge with existing data on disk."""
    # Atomic write with merge
    temp_file = output_file + '.tmp'
    try:
        # Load existing if available to merge
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    disk_data = json.load(f)
                
                # Merge 'results' dictionary recursively
                if 'results' in disk_data and 'results' in new_data:
                    for method, method_data in new_data['results'].items():
                        if method not in disk_data['results']:
                            disk_data['results'][method] = {}
                        for corruption, corruption_data in method_data.items():
                            if corruption not in disk_data['results'][method]:
                                disk_data['results'][method][corruption] = {}
                            for severity, result in corruption_data.items():
                                # Only update if we have a real result from this session
                                if result is not None:
                                    disk_data['results'][method][corruption][severity] = result
                
                # Update metadata and timestamp from the newest run
                disk_data['timestamp'] = new_data['timestamp']
                if 'summary' in new_data:
                    disk_data['summary'] = new_data['summary']
                
                final_data = disk_data
            except Exception as e:
                print(f"Warning: Could not merge with existing file: {e}")
                final_data = new_data
        else:
            final_data = new_data

        with open(temp_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        # Create backup
        if os.path.exists(output_file):
            shutil.copy2(output_file, output_file + '.backup')
            
        os.replace(temp_file, output_file)
    except Exception as e:
        print(f"Error saving results: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


def result_exists(results_dict, method, corruption, severity):
    """Check if a result already exists."""
    sev_key = str(severity)
    if method not in results_dict:
        return False
    if corruption not in results_dict[method]:
        return False
    if sev_key not in results_dict[method][corruption]:
        return False
    result = results_dict[method][corruption][sev_key]
    return result is not None and 'accuracy' in result


# ============================================================================
# Main Evaluation
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive robustness evaluation for ProtoLens')
    
    parser.add_argument('--data_dir', type=str, default='Datasets/Amazon-C')
    parser.add_argument('--model_path', type=str,
                       default='log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005/model.pth')
    parser.add_argument('--output', type=str, default='Datasets/Amazon-C/results/robustness_results.json')
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['baseline', 'tent', 'eata', 'prototta', 'sar'],
                       help='Methods to evaluate (additive - only adds missing)')
    parser.add_argument('--corruption_types', type=str, nargs='+', default=None)
    parser.add_argument('--severities', type=int, nargs='+', default=None)
    
    parser.add_argument('--learning_rate', type=float, default=0.000005)
    parser.add_argument('--adaptation_mode', type=str, default='layernorm_attn_bias')
    
    # EATA
    parser.add_argument('--e_margin', type=float, default=0.6)
    parser.add_argument('--d_margin', type=float, default=0.05)
    
    # ProtoTTA
    parser.add_argument('--geo_filter', action='store_true', default=True)
    parser.add_argument('--no_geo_filter', action='store_true', default=False)
    parser.add_argument('--geo_threshold', type=float, default=0.1)
    parser.add_argument('--sigmoid_temperature', type=float, default=5.0)
    parser.add_argument('--importance_mode', type=str, default='global')
    
    # Prototype metrics
    parser.add_argument('--prototype-metrics', action='store_true', default=True,
                       help='Compute prototype-based metrics')
    parser.add_argument('--no-prototype-metrics', action='store_true', default=False)
    parser.add_argument('--track-efficiency', action='store_true', default=True,
                       help='Track computational efficiency')
    
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--force', action='store_true', help='Force recompute existing results')
    parser.add_argument('--override', type=str, nargs='+', default=None,
                       help='Override specific methods (re-run and replace in JSON even if exists)')
    
    # Sharding for parallel evaluation
    parser.add_argument('--shard_id', type=int, default=0, help='Index of current shard (0 to num_shards-1)')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of parallel shards')
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Update cfg
    cfg.OPTIM.LR = args.learning_rate
    
    # Determine corruptions and severities
    if args.quick:
        corruption_types = QUICK_CORRUPTION_TYPES
        severities = QUICK_SEVERITIES
    else:
        corruption_types = args.corruption_types or ALL_CORRUPTION_TYPES
        severities = args.severities or ALL_SEVERITIES
    
    use_geo = args.geo_filter and not args.no_geo_filter
    
    print("=" * 80)
    print("ProtoLens Comprehensive Robustness Evaluation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Methods: {args.methods}")
    print(f"Corruptions: {corruption_types}")
    print(f"Severities: {severities}")
    print(f"Output: {args.output}")
    print(f"LR: {args.learning_rate}, Mode: {args.adaptation_mode}")
    print(f"ProtoTTA: geo_filter={use_geo}, geo_threshold={args.geo_threshold}, temp={args.sigmoid_temperature}")
    print("=" * 80)
    
    # Load existing results
    existing = load_results_json(args.output)
    
    if existing:
        results_data = existing
        results_dict = results_data.get('results', {})
        print(f"Loaded existing results from {args.output}")
    else:
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'model_path': args.model_path,
                'data_dir': args.data_dir,
                'batch_size': args.batch_size,
                'corruption_types': corruption_types,
                'severities': severities,
                'methods': args.methods,
                'learning_rate': args.learning_rate,
                'adaptation_mode': args.adaptation_mode,
                'mode_configs': {
                    'baseline': {},
                    'tent': {'learning_rate': args.learning_rate, 'adaptation_mode': args.adaptation_mode},
                    'eata': {'learning_rate': args.learning_rate, 'e_margin': args.e_margin,
                            'd_margin': args.d_margin, 'adaptation_mode': args.adaptation_mode},
                    'prototta': {'learning_rate': args.learning_rate, 'sigmoid_temperature': args.sigmoid_temperature,
                                'geo_filter': use_geo, 'geo_threshold': args.geo_threshold,
                                'importance_mode': args.importance_mode, 'adaptation_mode': args.adaptation_mode},
                    'sar': {'learning_rate': args.learning_rate, 'adaptation_mode': args.adaptation_mode}
                }
            },
            'results': {},
            'summary': {}
        }
        results_dict = results_data['results']
    
    # Initialize method entries
    for method in args.methods:
        if method not in results_dict:
            results_dict[method] = {}
    
    # Identify configurations (corruption, severity) that need work
    configs_needing_work = set()
    override_methods = set(args.override) if args.override else set()
    
    for method in args.methods:
        for corruption in corruption_types:
            for severity in severities:
                # Force recompute if: --force, --override includes this method, or result doesn't exist
                should_compute = (
                    args.force or 
                    (method in override_methods) or 
                    not result_exists(results_dict, method, corruption, severity)
                )
                if should_compute:
                    configs_needing_work.add((corruption, severity))
    
    # Sort and shard by CONFIG (not individual tasks)
    configs_to_process = sorted(list(configs_needing_work))
    
    if args.num_shards > 1:
        total_configs = len(configs_to_process)
        configs_to_process = [cfg for i, cfg in enumerate(configs_to_process) if i % args.num_shards == args.shard_id]
        print(f"Shard mode enabled: processing {len(configs_to_process)} of {total_configs} configs (Shard {args.shard_id}/{args.num_shards})")
    
    # For each config in this shard, determine which methods still need to run
    to_compute = []
    for corruption, severity in configs_to_process:
        for method in args.methods:
            should_compute = (
                args.force or 
                (method in override_methods) or 
                not result_exists(results_dict, method, corruption, severity)
            )
            if should_compute:
                to_compute.append((method, corruption, severity))
    
    print(f"\nTotal evaluations for this session: {len(to_compute)}")
    if len(to_compute) == 0:
        print("No evaluations left to run for this shard!")
        return
    
    # Track method accuracies for summary
    method_accuracies = {m: [] for m in args.methods}
    
    # Main evaluation loop
    eval_idx = 0
    
    # Group tasks by (corruption, severity) to avoid redundant data loading
    from collections import defaultdict
    tasks_by_config = defaultdict(list)
    for method, corruption, severity in to_compute:
        tasks_by_config[(corruption, severity)].append(method)
    
    # Sort configurations for consistent output
    sorted_configs = sorted(tasks_by_config.keys())
    
    for corruption, severity in sorted_configs:
        config_name = f"{corruption}_s{severity}"
        methods_to_run = tasks_by_config[(corruption, severity)]
        
        try:
            df = load_corrupted_data(args.data_dir, corruption, severity)
        except FileNotFoundError as e:
            print(f"\n[{config_name}] SKIPPED: {e}")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{config_name}] ({len(df)} samples) - Running shard tasks: {methods_to_run}")
        print(f"{'='*60}")
        
        # Get baseline accuracy for improvement calculation (if available)
        baseline_acc = None
        if 'baseline' in results_dict and corruption in results_dict['baseline']:
            sev_result = results_dict['baseline'][corruption].get(str(severity), {})
            if sev_result and 'accuracy' in sev_result:
                baseline_acc = sev_result['accuracy']
        
        # Prototype metrics and efficiency settings
        compute_proto_metrics = getattr(args, 'prototype_metrics', True) and not getattr(args, 'no_prototype_metrics', False)
        track_efficiency = getattr(args, 'track_efficiency', True)
        
        # Prepare a shared metrics evaluator for ALL methods in this config
        # Reference is the CLEAN (uncorrupted) data - this is the gold standard
        config_metrics_evaluator = None
        if compute_proto_metrics:
            print(f"  Collecting CLEAN baseline reference for PAC/PCA metrics...")
            try:
                clean_df = load_clean_data(args.data_dir)
                base_model, base_tokenizer, base_model_args = load_model(args.model_path, device)
                config_metrics_evaluator = ProtoLensMetricsEvaluator(base_model, device=str(device))
                clean_dataloader = create_dataloader(clean_df, base_tokenizer, base_model_args.max_length, args.batch_size)
                config_metrics_evaluator.collect_baseline(base_model, clean_dataloader, verbose=False)
                del base_model, clean_df
                torch.cuda.empty_cache()
                print(f"  Clean baseline collected successfully.")
            except FileNotFoundError as e:
                print(f"  Warning: Could not load clean data for metrics: {e}. PAC/PCA will be None.")
                config_metrics_evaluator = None
        
        for method in methods_to_run:
            eval_idx += 1
            print(f"\n--- [{eval_idx}/{len(to_compute)}] {method} on {config_name} ---")
            
            # Load fresh model for this method
            model, tokenizer, model_args = load_model(args.model_path, device)
            dataloader = create_dataloader(df, tokenizer, model_args.max_length, args.batch_size)
            
            # Setup and evaluate
            if method == 'baseline':
                # Baseline = unadapted model on CORRUPTED data
                # PAC/PCA compare this to the clean reference
                result = evaluate_baseline(
                    model, dataloader, device,
                    metrics_evaluator=config_metrics_evaluator,
                    track_efficiency=track_efficiency
                )
                baseline_acc = result['accuracy']
                
                # Compute PAC/PCA for baseline (unadapted on corrupted vs unadapted on clean)
                if config_metrics_evaluator is not None and result.get('_similarities') is not None:
                    adapted_activations = result.pop('_similarities')
                    labels_tensor = result.pop('_labels')
                    adapted_logits = result.pop('_logits')
                    adapted_predictions = result.pop('_predictions')
                    
                    result.update(config_metrics_evaluator.compute_pac(adapted_activations))
                    result.update(config_metrics_evaluator.compute_pca(adapted_activations, labels_tensor))
                    result.update(config_metrics_evaluator.compute_sparsity(adapted_activations))
                    result.update(config_metrics_evaluator.compute_pca_weighted(adapted_activations, labels_tensor))
                    result.update(config_metrics_evaluator.compute_calibration(adapted_predictions, adapted_logits))
                    result.update(config_metrics_evaluator.compute_gt_class_contribution(adapted_activations, labels_tensor))

                
            elif method in ['tent', 'eata', 'prototta', 'sar']:
                if method == 'tent':
                    tta_model = setup_tent(model, args.adaptation_mode)
                elif method == 'eata':
                    tta_model = setup_eata(model, args.adaptation_mode, args.e_margin, args.d_margin)
                elif method == 'prototta':
                    tta_model = setup_prototta(model, args.adaptation_mode, use_geo, args.geo_threshold,
                                              args.importance_mode, args.sigmoid_temperature)
                elif method == 'sar':
                    tta_model = setup_sar(model, args.adaptation_mode)
                
                result = evaluate_tta_method(
                    tta_model, dataloader, device, method,
                    metrics_evaluator=config_metrics_evaluator,
                    track_efficiency=track_efficiency,
                    compute_proto_metrics=compute_proto_metrics
                )
            
            else:
                print(f"  Unknown method: {method}")
                continue
            
            # Clean up internal tensor data before storing (not JSON serializable)
            for key in ['_similarities', '_labels', '_logits', '_predictions']:
                result.pop(key, None)
            
            # Store result
            if corruption not in results_dict[method]:
                results_dict[method][corruption] = {}
            results_dict[method][corruption][str(severity)] = result
            
            # Print summary
            acc = result['accuracy']
            if baseline_acc and method != 'baseline':
                improvement = (acc - baseline_acc) * 100
                imp_str = f"({improvement:+.2f}%)"
            else:
                imp_str = ""
            
            # Build metrics string
            metrics_str = ""
            if result.get('adaptation_rate') is not None and method != 'baseline':
                metrics_str += f" adapt:{result['adaptation_rate']*100:.1f}%"
            if result.get('PAC_mean') is not None:
                metrics_str += f" PAC:{result['PAC_mean']*100:.1f}%"
            if result.get('PCA_mean') is not None:
                metrics_str += f" PCA:{result['PCA_mean']*100:.1f}%"
                
            print(f"  Result: {acc:.4f} {imp_str}{metrics_str}")
            
            # Save incrementally
            results_data['timestamp'] = datetime.now().isoformat()
            results_data['results'] = results_dict
            save_results_json(args.output, results_data)
            print(f"  Saved to {args.output}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()


    
    # Compute summary statistics
    print("\n" + "=" * 80)
    print("COMPUTING SUMMARY STATISTICS")
    print("=" * 80)
    
    summary = {}
    baseline_accs = []
    
    for method in args.methods:
        if method in results_dict:
            accs = []
            for corruption in corruption_types:
                if corruption in results_dict[method]:
                    for sev in severities:
                        result = results_dict[method][corruption].get(str(sev), {})
                        if result and 'accuracy' in result:
                            accs.append(result['accuracy'])
                            if method == 'baseline':
                                baseline_accs.append(result['accuracy'])
            
            if accs:
                summary[method] = {
                    'avg_accuracy': float(np.mean(accs)),
                    'std_accuracy': float(np.std(accs)),
                    'min_accuracy': float(np.min(accs)),
                    'max_accuracy': float(np.max(accs)),
                    'num_evaluations': len(accs)
                }
    
    # Add improvement over baseline
    if baseline_accs:
        baseline_avg = np.mean(baseline_accs)
        for method in summary:
            if method != 'baseline':
                summary[method]['improvement_over_baseline'] = float(
                    (summary[method]['avg_accuracy'] - baseline_avg) * 100
                )
    
    results_data['summary'] = summary
    save_results_json(args.output, results_data)
    
    # Print summary table
    print(f"\n{'Method':<16} {'Avg Acc':<12} {'Std':<10} {'Improvement':<12}")
    print("-" * 55)
    for method, stats in summary.items():
        imp = stats.get('improvement_over_baseline', 0)
        imp_str = f"{imp:+.2f}%" if method != 'baseline' else ""
        print(f"{method:<16} {stats['avg_accuracy']:.4f}       {stats['std_accuracy']:.4f}     {imp_str}")
    
    print("\n" + "=" * 80)
    print(f"EVALUATION COMPLETE - Results saved to {args.output}")
    print("=" * 80)


if __name__ == '__main__':
    main()
