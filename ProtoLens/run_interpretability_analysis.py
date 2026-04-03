"""
Run interpretability analysis for ProtoLens on selected samples.

This script:
1. Loads models with different TTA methods
2. Selects samples where baseline fails but ProtoTTA succeeds
3. Creates comprehensive visualizations showing prototype activations
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import random

# Import ProtoLens modules
from PLens import BERTClassifier
from utils import TextClassificationDataset
import tent
import eata
import proto_tta
import adapt_utils
import interpretability_viz

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class Cfg:
    """Configuration for TTA optimization."""
    def __init__(self):
        self.OPTIM = self.Optim()
        self.MODEL = self.Model()

    class Optim:
        def __init__(self):
            self.METHOD = 'Adam'
            self.LR = 0.000005
            self.BETA = 0.9
            self.WD = 0.0
            self.STEPS = 1

    class Model:
        def __init__(self):
            self.EPISODIC = False

cfg = Cfg()


def load_model(model_path, device):
    """Load ProtoLens model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})
    
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


def setup_optimizer(params):
    """Setup optimizer for TTA."""
    if cfg.OPTIM.METHOD == 'Adam':
        return torch.optim.Adam(params, lr=cfg.OPTIM.LR, betas=(cfg.OPTIM.BETA, 0.999),
                               weight_decay=cfg.OPTIM.WD)
    else:
        return torch.optim.SGD(params, lr=cfg.OPTIM.LR, momentum=0.9, weight_decay=cfg.OPTIM.WD)


def setup_tent(model, adaptation_mode):
    """Setup TENT adaptation."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    optimizer = setup_optimizer(params)
    return tent.Tent(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def setup_eata(model, adaptation_mode, e_margin, d_margin):
    """Setup EATA adaptation."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    optimizer = setup_optimizer(params)
    return eata.EATA(model, optimizer, fishers=None, steps=cfg.OPTIM.STEPS,
                     episodic=cfg.MODEL.EPISODIC, e_margin=e_margin, d_margin=d_margin)


def setup_prototta(model, adaptation_mode, use_geo_filter, geo_threshold,
                   importance_mode='global', sigmoid_temperature=5.0):
    """Setup ProtoTTA adaptation."""
    model = adapt_utils.configure_model(model, adaptation_mode)
    params, param_names = adapt_utils.collect_params(model, adaptation_mode)
    optimizer = setup_optimizer(params)
    return proto_tta.ProtoTTA(model, optimizer, steps=cfg.OPTIM.STEPS,
                              episodic=cfg.MODEL.EPISODIC,
                              use_geometric_filter=use_geo_filter,
                              geo_filter_threshold=geo_threshold,
                              consensus_strategy='max',
                              importance_mode=importance_mode,
                              sigmoid_temperature=sigmoid_temperature)


def evaluate_single_sample(model, text, tokenizer, max_length, device):
    """Evaluate a single text sample and return prediction and similarities.
    
    Uses DataLoader to ensure proper batch handling.
    """
    from torch.utils.data import DataLoader
    
    # Create a dataset with a single sample
    dataset = TextClassificationDataset([text], [0], tokenizer, max_length)  # dummy label
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)
            
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mode="test",
                original_text=batch['original_text'],
                current_batch_num=0
            )
            
            if isinstance(result, tuple) and len(result) >= 4:
                logits = result[0]
                similarities = result[3]
                # Handle different similarity tensor shapes
                if similarities.dim() == 3:
                    # [batch, seq_len, num_prototypes] -> [batch, num_prototypes]
                    similarities = similarities.mean(dim=1)
                elif similarities.dim() == 1:
                    # [num_prototypes] -> [1, num_prototypes]
                    similarities = similarities.unsqueeze(0)
                # similarities should now be [1, num_prototypes]
            else:
                logits = result[0] if isinstance(result, tuple) else result
                similarities = None
            
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, predicted_class].item()
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'logits': logits,
                'similarities': similarities[0] if similarities is not None else None
            }
    
    # Should not reach here
    return None


def find_interesting_samples(clean_df, corrupted_df, baseline_model, prototta_model,
                            tokenizer, max_length, device, num_samples=5):
    """
    Find samples where:
    - Baseline works on clean (correct prediction)
    - Baseline fails on corrupted (wrong prediction)
    - ProtoTTA succeeds on corrupted (correct prediction)
    """
    interesting_samples = []
    
    print("Searching for interesting samples...")
    for idx in tqdm(range(min(len(corrupted_df), 1000)), desc="Scanning samples"):
        clean_text = clean_df.iloc[idx]['review']
        corrupted_text = corrupted_df.iloc[idx]['review']
        true_label = corrupted_df.iloc[idx]['sentiment']
        
        # Evaluate baseline on clean
        baseline_clean = evaluate_single_sample(baseline_model, clean_text, 
                                               tokenizer, max_length, device)
        
        # Evaluate baseline on corrupted
        baseline_corrupted = evaluate_single_sample(baseline_model, corrupted_text,
                                                     tokenizer, max_length, device)
        
        # Evaluate ProtoTTA on corrupted
        prototta_corrupted = evaluate_single_sample(prototta_model, corrupted_text,
                                                    tokenizer, max_length, device)
        
        # Check if this is an interesting sample
        baseline_clean_correct = (baseline_clean['predicted_class'] == true_label)
        baseline_corrupted_wrong = (baseline_corrupted['predicted_class'] != true_label)
        prototta_corrupted_correct = (prototta_corrupted['predicted_class'] == true_label)
        
        if baseline_clean_correct and baseline_corrupted_wrong and prototta_corrupted_correct:
            interesting_samples.append({
                'index': idx,
                'clean_text': clean_text,
                'corrupted_text': corrupted_text,
                'true_label': true_label,
                'baseline_clean_pred': baseline_clean['predicted_class'],
                'baseline_corrupted_pred': baseline_corrupted['predicted_class'],
                'prototta_corrupted_pred': prototta_corrupted['predicted_class']
            })
            
            if len(interesting_samples) >= num_samples:
                break
    
    return interesting_samples


def main():
    parser = argparse.ArgumentParser(description='Run interpretability analysis for ProtoLens')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained ProtoLens model')
    parser.add_argument('--clean_data', type=str, 
                       default='Datasets/Amazon/test.csv',
                       help='Path to clean test data')
    parser.add_argument('--corrupted_data', type=str, required=True,
                       help='Path to corrupted data (e.g., Datasets/Amazon-C/amazon_c_qwerty_s40.csv)')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to analyze')
    parser.add_argument('--corruption_name', type=str, default='qwerty',
                       help='Name of corruption type')
    parser.add_argument('--severity', type=int, default=40,
                       help='Corruption severity')
    parser.add_argument('--learning_rate', type=float, default=0.000005)
    parser.add_argument('--adaptation_mode', type=str, default='layernorm_attn_bias')
    parser.add_argument('--geo_filter', action='store_true', default=True)
    parser.add_argument('--geo_threshold', type=float, default=0.1)
    parser.add_argument('--e_margin', type=float, default=0.6)
    parser.add_argument('--d_margin', type=float, default=0.05)
    parser.add_argument('--sigmoid_temperature', type=float, default=5.0)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.OPTIM.LR = args.learning_rate
    
    print("="*80)
    print("ProtoLens Interpretability Analysis")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Clean data: {args.clean_data}")
    print(f"Corrupted data: {args.corrupted_data}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer, model_args = load_model(args.model_path, device)
    max_length = model_args.max_length
    
    # Load data
    print("\nLoading data...")
    clean_df = pd.read_csv(args.clean_data)
    corrupted_df = pd.read_csv(args.corrupted_data)
    
    # Setup models for different methods
    print("\nSetting up adaptation methods...")
    models_dict = {}
    
    # Baseline (no adaptation)
    baseline_model = model
    models_dict['Baseline'] = baseline_model
    
    # TENT
    tent_model = setup_tent(model, args.adaptation_mode)
    models_dict['TENT'] = tent_model
    
    # EATA
    eata_model = setup_eata(model, args.adaptation_mode, args.e_margin, args.d_margin)
    models_dict['EATA'] = eata_model
    
    # ProtoTTA
    prototta_model = setup_prototta(model, args.adaptation_mode, args.geo_filter,
                                    args.geo_threshold, sigmoid_temperature=args.sigmoid_temperature)
    models_dict['ProtoTTA'] = prototta_model
    
    # Find interesting samples
    print("\nFinding interesting samples...")
    interesting_samples = find_interesting_samples(
        clean_df, corrupted_df, baseline_model, prototta_model,
        tokenizer, max_length, device, num_samples=args.num_samples
    )
    
    if len(interesting_samples) == 0:
        print("⚠ No interesting samples found. Using random samples instead...")
        # Fallback to random samples
        random_indices = np.random.choice(len(corrupted_df), size=min(args.num_samples, len(corrupted_df)), replace=False)
        interesting_samples = []
        for idx in random_indices:
            interesting_samples.append({
                'index': idx,
                'clean_text': clean_df.iloc[idx]['review'],
                'corrupted_text': corrupted_df.iloc[idx]['review'],
                'true_label': corrupted_df.iloc[idx]['sentiment']
            })
    
    print(f"\n✓ Found {len(interesting_samples)} samples to analyze")
    
    # Run interpretability analysis for each sample
    experimental_settings = {
        'model_path': args.model_path,
        'corruption': args.corruption_name,
        'severity': args.severity,
        'methods': list(models_dict.keys()),
        'learning_rate': args.learning_rate,
        'adaptation_mode': args.adaptation_mode,
        'geo_filter': args.geo_filter,
        'geo_threshold': args.geo_threshold,
        'e_margin': args.e_margin,
        'd_margin': args.d_margin,
        'sigmoid_temperature': args.sigmoid_temperature
    }
    
    output_dirs = []
    for i, sample in enumerate(interesting_samples):
        print(f"\n{'='*60}")
        print(f"Analyzing sample {i+1}/{len(interesting_samples)}")
        print(f"{'='*60}")
        
        try:
            output_dir = interpretability_viz.run_comprehensive_interpretability(
                models_dict=models_dict,
                clean_text=sample['clean_text'],
                corrupted_text=sample['corrupted_text'],
                tokenizer=tokenizer,
                max_length=max_length,
                output_base_dir=args.output_dir,
                corruption_name=args.corruption_name,
                severity=args.severity,
                experimental_settings=experimental_settings,
                true_label=sample['true_label'],
                sample_id=f"sample_{sample['index']}"
            )
            output_dirs.append(output_dir)
        except Exception as e:
            print(f"⚠ Error analyzing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Interpretability analysis complete!")
    print(f"Generated {len(output_dirs)} visualization sets")
    for out_dir in output_dirs:
        print(f"  - {out_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
