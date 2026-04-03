"""
Comprehensive Interpretability Visualization for ProtoLens TTA Methods

This module creates detailed visualizations showing:
1. Original clean text + corrupted text
2. Prototype similarities for clean text (baseline model)
3. Prototype similarities for corrupted text (baseline, TENT, EATA, ProtoTTA)
4. Predictions and confidence scores for each method
5. Experimental settings documentation

File organization:
plots/interpretability_comprehensive/{corruption}_{severity}/
    {sample_id}/
        00_original_clean_text.txt
        01_corrupted_text.txt
        02_Baseline_CLEAN_analysis.png
        03_Baseline_CORRUPTED_analysis.png
        03_TENT_CORRUPTED_analysis.png
        03_EATA_CORRUPTED_analysis.png
        03_ProtoTTA_CORRUPTED_analysis.png
        04_prototype_similarities_comparison.png
        05_comparison_summary.txt
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score
import textwrap

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150


def makedir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def wrap_text(text, width=80):
    """Wrap text for display."""
    return '\n'.join(textwrap.wrap(text, width=width))


def get_top_k_prototypes(similarities, fc_weights, predicted_class, k=10, 
                         sort_by='similarity'):
    """
    Get top-k prototypes by similarity or contribution.
    
    Args:
        similarities: [num_prototypes] tensor of prototype similarities
        fc_weights: [num_classes, num_prototypes] tensor of FC layer weights
        predicted_class: int, predicted class index
        k: number of top prototypes to return
        sort_by: 'similarity' or 'contribution' (similarity * weight)
    
    Returns:
        List of dicts with prototype info
    """
    similarities_np = similarities.detach().cpu().numpy()
    weights_np = fc_weights.detach().cpu().numpy()
    
    proto_results = []
    for proto_idx in range(len(similarities_np)):
        similarity = similarities_np[proto_idx]
        weight = weights_np[predicted_class, proto_idx]
        contribution = similarity * max(0, weight)  # Only positive weights contribute
        
        proto_results.append({
            'proto_idx': proto_idx,
            'similarity': float(similarity),
            'weight': float(weight),
            'contribution': float(contribution)
        })
    
    # Sort by requested metric
    if sort_by == 'contribution':
        proto_results.sort(key=lambda x: x['contribution'], reverse=True)
    else:  # similarity
        proto_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return proto_results[:k]


def create_text_visualization(text, title, max_length=500):
    """Create a text visualization with word wrapping."""
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    wrapped = wrap_text(text, width=80)
    return wrapped


def create_prototype_similarity_plot(proto_results, method_name, top_k=10):
    """Create a bar plot showing top-k prototype similarities."""
    if len(proto_results) == 0:
        return None
    
    top_protos = proto_results[:top_k]
    proto_indices = [p['proto_idx'] for p in top_protos]
    similarities = [p['similarity'] for p in top_protos]
    contributions = [p['contribution'] for p in top_protos]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Similarities
    colors = plt.cm.viridis(np.linspace(0, 1, len(proto_indices)))
    bars1 = ax1.barh(range(len(proto_indices)), similarities, color=colors)
    ax1.set_yticks(range(len(proto_indices)))
    ax1.set_yticklabels([f'Proto {idx}' for idx in proto_indices])
    ax1.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'{method_name}\nTop-{top_k} Prototype Similarities', 
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, similarities)):
        ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Plot 2: Contributions
    bars2 = ax2.barh(range(len(proto_indices)), contributions, color=colors)
    ax2.set_yticks(range(len(proto_indices)))
    ax2.set_yticklabels([f'Proto {idx}' for idx in proto_indices])
    ax2.set_xlabel('Contribution (Similarity × Weight)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{method_name}\nTop-{top_k} Prototype Contributions', 
                  fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, contributions)):
        if val > 0:
            ax2.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_comprehensive_analysis(model_or_wrapper, text, corrupted_text, 
                                  method_name, tokenizer, max_length,
                                  true_label=None, precomputed_result=None):
    """
    Create comprehensive analysis visualization for a single method.
    
    Returns:
        fig: matplotlib figure
        proto_results: list of prototype results
        predicted_class: int
        confidence: float
    """
    # Extract model
    if hasattr(model_or_wrapper, 'model'):
        model = model_or_wrapper.model
    else:
        model = model_or_wrapper
    
    model.eval()
    
    # Prepare input
    encoded = tokenizer(
        [text if 'Clean' in method_name else corrupted_text],
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(next(model.parameters()).device)
    attention_mask = encoded['attention_mask'].to(next(model.parameters()).device)
    special_tokens_mask = encoded.get('special_tokens_mask', 
                                      torch.zeros_like(input_ids)).to(next(model.parameters()).device)
    
    # Forward pass
    with torch.no_grad():
        if precomputed_result is not None:
            logits = precomputed_result['logits']
            similarities = precomputed_result['similarities']
        else:
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask,
                mode="test",
                original_text=[text if 'Clean' in method_name else corrupted_text],
                current_batch_num=0
            )
            if isinstance(result, tuple) and len(result) >= 4:
                logits = result[0]
                similarities = result[3]
            else:
                logits = result[0] if isinstance(result, tuple) else result
                similarities = None
    
    # Get prediction
    probs = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item()
    
    # Get FC weights
    fc_weights = model.fc.weight.data  # [num_classes, num_prototypes]
    
    # Ensure similarities is 2D
    if similarities is not None:
        if similarities.dim() == 3:
            similarities = similarities.mean(dim=1)  # Average over sequence
        if similarities.dim() == 1:
            similarities = similarities.unsqueeze(0)
        similarities = similarities[0]  # Get first (and only) sample
    else:
        # Fallback: create dummy similarities
        num_prototypes = fc_weights.shape[1]
        similarities = torch.zeros(num_prototypes, device=fc_weights.device)
    
    # Get top prototypes
    proto_results = get_top_k_prototypes(
        similarities, fc_weights, predicted_class, k=10, sort_by='contribution'
    )
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Text display (top left)
    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis('off')
    display_text = text if 'Clean' in method_name else corrupted_text
    wrapped_text = wrap_text(display_text, width=100)
    ax_text.text(0.05, 0.95, wrapped_text, transform=ax_text.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax_text.set_title(f'{method_name}\nText Sample', fontsize=14, fontweight='bold')
    
    # Prediction info (top right)
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_pred.axis('off')
    pred_text = f"Predicted Class: {predicted_class}\n"
    pred_text += f"Confidence: {confidence:.4f}\n"
    if true_label is not None:
        pred_text += f"True Label: {true_label}\n"
        is_correct = "✓ CORRECT" if predicted_class == true_label else "✗ WRONG"
        pred_text += f"Result: {is_correct}"
    
    ax_pred.text(0.1, 0.5, pred_text, transform=ax_pred.transAxes,
                fontsize=14, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if (true_label is None or predicted_class == true_label) else 'lightcoral', alpha=0.7))
    ax_pred.set_title('Prediction', fontsize=14, fontweight='bold')
    
    # Prototype similarities bar plot (bottom left)
    ax_sim = fig.add_subplot(gs[1:, 0])
    if len(proto_results) > 0:
        top_protos = proto_results[:10]
        proto_indices = [p['proto_idx'] for p in top_protos]
        similarities_vals = [p['similarity'] for p in top_protos]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(proto_indices)))
        bars = ax_sim.barh(range(len(proto_indices)), similarities_vals, color=colors)
        ax_sim.set_yticks(range(len(proto_indices)))
        ax_sim.set_yticklabels([f'Proto {idx}' for idx in proto_indices])
        ax_sim.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        ax_sim.set_title('Top-10 Prototype Similarities', fontsize=12, fontweight='bold')
        ax_sim.grid(axis='x', alpha=0.3)
        ax_sim.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, similarities_vals)):
            ax_sim.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    # Prototype contributions bar plot (bottom right)
    ax_contrib = fig.add_subplot(gs[1:, 1])
    if len(proto_results) > 0:
        top_protos = proto_results[:10]
        proto_indices = [p['proto_idx'] for p in top_protos]
        contributions = [p['contribution'] for p in top_protos]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(proto_indices)))
        bars = ax_contrib.barh(range(len(proto_indices)), contributions, color=colors)
        ax_contrib.set_yticks(range(len(proto_indices)))
        ax_contrib.set_yticklabels([f'Proto {idx}' for idx in proto_indices])
        ax_contrib.set_xlabel('Contribution (Similarity × Weight)', fontsize=12, fontweight='bold')
        ax_contrib.set_title('Top-10 Prototype Contributions', fontsize=12, fontweight='bold')
        ax_contrib.grid(axis='x', alpha=0.3)
        ax_contrib.invert_yaxis()
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, contributions)):
            if val > 0:
                ax_contrib.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
    
    plt.suptitle(f'{method_name} Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    return fig, proto_results, predicted_class, confidence


def create_similarity_comparison_heatmap(models_dict, clean_text, corrupted_text,
                                         tokenizer, max_length, image_output_dir):
    """Create a heatmap comparing prototype similarities across methods."""
    num_methods = len(models_dict) + 1  # +1 for clean baseline
    num_prototypes = None
    
    # First pass: get number of prototypes
    for model_wrapper in models_dict.values():
        if hasattr(model_wrapper, 'model'):
            model = model_wrapper.model
        else:
            model = model_wrapper
        if hasattr(model, 'fc'):
            num_prototypes = model.fc.weight.shape[1]
            break
    
    if num_prototypes is None:
        print("⚠ Could not determine number of prototypes")
        return None
    
    # Collect similarities for each method
    all_similarities = {}
    
    # Clean baseline
    if 'Baseline' in models_dict:
        model_wrapper = models_dict['Baseline']
        model = model_wrapper.model if hasattr(model_wrapper, 'model') else model_wrapper
        model.eval()
        
        encoded = tokenizer([clean_text], max_length=max_length, padding='max_length',
                           truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(next(model.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(model.parameters()).device)
        special_tokens_mask = encoded.get('special_tokens_mask', 
                                         torch.zeros_like(input_ids)).to(next(model.parameters()).device)
        
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask,
                          special_tokens_mask=special_tokens_mask, mode="test",
                          original_text=[clean_text], current_batch_num=0)
            if isinstance(result, tuple) and len(result) >= 4:
                similarities = result[3]
                if similarities.dim() == 3:
                    similarities = similarities.mean(dim=1)
                if similarities.dim() == 1:
                    similarities = similarities.unsqueeze(0)
                all_similarities['Baseline (Clean)'] = similarities[0].detach().cpu().numpy()
    
    # Corrupted with each method
    for method_name, model_wrapper in models_dict.items():
        if hasattr(model_wrapper, 'model'):
            model = model_wrapper.model
        else:
            model = model_wrapper
        model.eval()
        
        encoded = tokenizer([corrupted_text], max_length=max_length, padding='max_length',
                           truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(next(model.parameters()).device)
        attention_mask = encoded['attention_mask'].to(next(model.parameters()).device)
        special_tokens_mask = encoded.get('special_tokens_mask', 
                                         torch.zeros_like(input_ids)).to(next(model.parameters()).device)
        
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask,
                          special_tokens_mask=special_tokens_mask, mode="test",
                          original_text=[corrupted_text], current_batch_num=0)
            if isinstance(result, tuple) and len(result) >= 4:
                similarities = result[3]
                if similarities.dim() == 3:
                    similarities = similarities.mean(dim=1)
                if similarities.dim() == 1:
                    similarities = similarities.unsqueeze(0)
                all_similarities[method_name] = similarities[0].detach().cpu().numpy()
    
    # Create heatmap
    if len(all_similarities) == 0:
        return None
    
    # Stack similarities
    similarity_matrix = np.array([all_similarities[name] for name in all_similarities.keys()])
    
    # Show top-k prototypes (by max similarity across methods)
    max_similarities = similarity_matrix.max(axis=0)
    top_k = min(20, num_prototypes)
    top_indices = np.argsort(max_similarities)[-top_k:][::-1]
    
    # Filter to top-k
    similarity_matrix_filtered = similarity_matrix[:, top_indices]
    proto_labels = [f'Proto {idx}' for idx in top_indices]
    
    fig, ax = plt.subplots(figsize=(max(12, top_k * 0.5), max(6, len(all_similarities) * 0.8)))
    im = ax.imshow(similarity_matrix_filtered, cmap='viridis', aspect='auto')
    
    ax.set_xticks(range(len(proto_labels)))
    ax.set_xticklabels(proto_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(all_similarities)))
    ax.set_yticklabels(list(all_similarities.keys()))
    ax.set_title('Prototype Similarities Comparison\n(Top-20 Prototypes)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity Score', fontsize=12)
    
    # Add text annotations
    for i in range(len(all_similarities)):
        for j in range(len(proto_labels)):
            text = ax.text(j, i, f'{similarity_matrix_filtered[i, j]:.2f}',
                          ha="center", va="center", color="white" if similarity_matrix_filtered[i, j] < 0.5 else "black",
                          fontsize=8)
    
    plt.tight_layout()
    
    heatmap_path = os.path.join(image_output_dir, '04_prototype_similarities_comparison.png')
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return heatmap_path


def run_comprehensive_interpretability(
    models_dict,  # Dict: {method_name: model_wrapper}
    clean_text,   # Original clean text
    corrupted_text,  # Corrupted text
    tokenizer,
    max_length,
    output_base_dir,
    corruption_name,
    severity,
    experimental_settings,
    true_label=None,
    sample_id=None,
    precomputed_results_dict=None
):
    """
    Create comprehensive interpretability visualizations for multiple TTA methods.
    """
    # Create output directory
    corruption_str = f"{corruption_name}_sev{severity}" if corruption_name else "clean"
    if sample_id is None:
        sample_id = f"sample_{hash(corrupted_text) % 10000}"
    
    image_output_dir = os.path.join(
        output_base_dir,
        'interpretability_comprehensive',
        corruption_str,
        str(sample_id)
    )
    makedir(image_output_dir)
    
    print(f"\n{'='*60}")
    print(f"Interpretability: Sample {sample_id}")
    print(f"Output: {image_output_dir}")
    print(f"{'='*60}\n")
    
    # Save texts
    with open(os.path.join(image_output_dir, '00_original_clean_text.txt'), 'w') as f:
        f.write(clean_text)
    
    with open(os.path.join(image_output_dir, '01_corrupted_text.txt'), 'w') as f:
        f.write(corrupted_text)
    
    # Analyze CLEAN text with baseline
    all_proto_results = {}
    all_predictions = {}
    
    if 'Baseline' in models_dict:
        print("Analyzing CLEAN text with Baseline model...")
        model_wrapper = models_dict['Baseline']
        
        precomputed = None
        if precomputed_results_dict and 'Baseline_Clean' in precomputed_results_dict:
            precomputed = precomputed_results_dict['Baseline_Clean']
        
        fig, proto_results, pred_class, confidence = create_comprehensive_analysis(
            model_wrapper, clean_text, corrupted_text, 'Baseline (Clean)',
            tokenizer, max_length, true_label=true_label, precomputed_result=precomputed
        )
        
        clean_analysis_path = os.path.join(image_output_dir, '02_Baseline_CLEAN_analysis.png')
        fig.savefig(clean_analysis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved clean baseline: {clean_analysis_path}")
        print(f"  Predicted class: {pred_class}, Confidence: {confidence:.4f}")
        all_predictions['Baseline (Clean)'] = {'class': pred_class, 'confidence': confidence}
    
    # Analyze CORRUPTED text with all methods
    for method_name, model_wrapper in models_dict.items():
        print(f"\nAnalyzing CORRUPTED text with {method_name}...")
        
        precomputed = None
        if precomputed_results_dict and method_name in precomputed_results_dict:
            precomputed = precomputed_results_dict[method_name]
            print(f"  Using PRECOMPUTED results for {method_name}")
        
        fig, proto_results, pred_class, confidence = create_comprehensive_analysis(
            model_wrapper, clean_text, corrupted_text, f'{method_name} (Corrupted)',
            tokenizer, max_length, true_label=true_label, precomputed_result=precomputed
        )
        
        analysis_path = os.path.join(
            image_output_dir,
            f'03_{method_name.replace(" ", "_")}_CORRUPTED_analysis.png'
        )
        fig.savefig(analysis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved: {analysis_path}")
        print(f"  Predicted class: {pred_class}, Confidence: {confidence:.4f}")
        
        all_proto_results[method_name] = proto_results
        all_predictions[method_name] = {'class': pred_class, 'confidence': confidence}
    
    # Create similarity comparison heatmap
    print(f"\nCreating prototype similarities comparison...")
    try:
        heatmap_path = create_similarity_comparison_heatmap(
            models_dict, clean_text, corrupted_text, tokenizer, max_length, image_output_dir
        )
        if heatmap_path:
            print(f"✓ Saved similarity heatmap: {heatmap_path}")
    except Exception as e:
        print(f"⚠ Could not create heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    # Save comparison summary
    summary_path = os.path.join(image_output_dir, '05_comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PROTOTYPE SIMILARITY COMPARISON\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Sample ID: {sample_id}\n")
        f.write(f"Corruption: {corruption_name or 'None'}\n")
        if corruption_name:
            f.write(f"Severity: {severity}\n")
        if true_label is not None:
            f.write(f"True Label: {true_label}\n")
        f.write(f"\nMethods: {', '.join(models_dict.keys())}\n\n")
        
        f.write("="*60 + "\n")
        f.write("PREDICTIONS PER METHOD\n")
        f.write("="*60 + "\n\n")
        
        for method_name, pred_info in all_predictions.items():
            is_correct = "✓" if (true_label is not None and pred_info['class'] == true_label) else "✗"
            f.write(f"{method_name}:\n")
            f.write(f"  {is_correct} Predicted: {pred_info['class']}, "
                   f"Confidence: {pred_info['confidence']:.4f}\n")
            if true_label is not None:
                f.write(f"  True Label: {true_label}\n")
            f.write("\n")
        
        f.write("="*60 + "\n")
        f.write("TOP-5 PROTOTYPES PER METHOD\n")
        f.write("="*60 + "\n\n")
        
        for method_name, proto_results in all_proto_results.items():
            f.write(f"\n{method_name}:\n")
            f.write("-"*60 + "\n")
            for i, proto in enumerate(proto_results[:5]):
                f.write(f"  {i+1}. Proto {proto['proto_idx']:4d} | "
                       f"Similarity: {proto['similarity']:6.4f} | "
                       f"Weight: {proto['weight']:7.4f} | "
                       f"Contribution: {proto['contribution']:7.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*60 + "\n\n")
        f.write("This visualization shows:\n")
        f.write("  1. How each method processes the corrupted text\n")
        f.write("  2. Which prototypes are most activated (similarity)\n")
        f.write("  3. Which prototypes contribute most to the prediction (contribution)\n")
        f.write("  4. How adaptation affects prototype activations\n\n")
        f.write("Key observations:\n")
        f.write("  - Higher similarity = stronger match to prototype\n")
        f.write("  - Contribution = similarity × weight (only positive weights)\n")
        f.write("  - Adaptation should improve prototype activations for correct class\n")
    
    print(f"✓ Saved comparison summary: {summary_path}")
    
    return image_output_dir
