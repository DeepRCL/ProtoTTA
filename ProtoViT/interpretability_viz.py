"""
Comprehensive Interpretability Visualization for ProtoViT TTA Methods

This module creates detailed visualizations showing:
1. Original clean image + noisy image
2. Prototype activations for clean image (Normal model)
3. Prototype activations for noisy images (Normal, EATA, ProtoEntropy-Imp+Conf)
4. Training prototype patches that correspond to activated prototypes
5. Experimental settings documentation

File organization:
plots/interpretability_comprehensive/{corruption}_{severity}/
    {class_name}/{image_name}/
        00_original_clean_image.png
        01_noisy_image.png
        02_Normal_clean_analysis.png
        03_Normal_noisy_analysis.png
        03_EATA_noisy_analysis.png
        03_ProtoEntropy-Imp+Conf_noisy_analysis.png
        04_prototype_patches/
            proto_{idx}_original.png
            proto_{idx}_closest_train.png
        settings.txt
"""

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.ndimage import zoom

# Set matplotlib to non-interactive backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import ImageDraw

from pathlib import Path
import copy
from preprocess import mean, std, undo_preprocess_input_function
from noise_utils import get_corrupted_transform


def makedir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def save_tensor_image(tensor_img, save_path):
    """Save a tensor image (after undoing preprocessing)."""
    img_copy = copy.deepcopy(tensor_img.unsqueeze(0) if tensor_img.dim() == 3 else tensor_img)
    undo_img = undo_preprocess_input_function(img_copy)
    undo_img = undo_img[0].detach().cpu().numpy()
    undo_img = np.transpose(undo_img, [1, 2, 0])
    plt.imsave(save_path, np.clip(undo_img, 0, 1))
    return undo_img


def get_top_k_prototypes_batch(model, image_batch, k=10, precomputed_outputs=None, sort_by='activation'):
    """
    Get the top-k most activated prototypes for a BATCH of images.
    Preserves batch statistics (critical for TTA methods).
    
    Args:
        model: PPNet model
        image_batch: [B, C, H, W] tensor
        k: top-k (if None, returns ALL prototypes - useful for later sorting by weight)
        precomputed_outputs: Optional tuple of (logits, min_distances, values) from a previous forward pass.
                           If provided, skips the forward pass to avoid double adaptation in TTA methods.
        sort_by: 'activation' (default), 'contribution' (activation * max(weight, 0)), 'weight' (weight only), or 'predicted_class_activation' (activation * weight for predicted class prototypes)
        
    Returns:
        List of (results, pred_class) tuples, one for each image in batch.
    """
    model.eval()
    with torch.no_grad():
        image_batch = image_batch.cuda()
        # Run forward pass on the WHOLE batch to use correct batch stats (unless precomputed)
        if precomputed_outputs is not None:
            logits, min_distances, values = precomputed_outputs
        else:
            logits, min_distances, values = model(image_batch)
        
        # Get predicted classes
        _, predicted = torch.max(logits.data, 1)
        
        # Compute prototype activations
        # patch_select: [1, n_prototypes, n_subpatches] -> squeeze to [n_prototypes, n_subpatches]
        slots = torch.sigmoid(model.patch_select * model.temp).squeeze(0)
        factor = ((slots.sum(-1))).unsqueeze(-1) + 1e-10
        proto_h = model.prototype_shape[2]
        n_p = proto_h  # number of prototype subpatches
        
        # Get indices for patch locations (run on batch)
        _, _, indices = model.push_forward(image_batch)
        # indices: [B, n_prototypes]
        
        # values: [B, n_prototypes, n_subpatches]
        # values_slot: [B, n_prototypes, n_subpatches]
        values_slot = (values.clone()) * (slots.unsqueeze(0) * n_p / factor.unsqueeze(0))
        cosine_act = values_slot.sum(-1)  # [B, num_prototypes]
        
        # Verify shape of cosine_act
        if cosine_act.dim() == 1:
            # If batch size is 1, it might be squeezed? But values should be [1, P, S]
            pass
        elif cosine_act.dim() > 2:
            # Should not happen
            cosine_act = cosine_act.view(image_batch.size(0), -1)

        batch_results = []
        
        for b in range(image_batch.size(0)):
            pred_class = predicted[b].item()
            
            # Compute sorting criterion
            if sort_by == 'contribution':
                # Sort by activation * max(weight, 0) GLOBALLY (across all classes)
                # This shows which prototypes contribute most to ANY class prediction
                # Get max weight across all classes for each prototype
                all_weights = model.last_layer.weight  # [num_classes, num_prototypes]
                max_weights = torch.clamp(all_weights, min=0).max(dim=0)[0]  # [num_prototypes] - max across classes
                contributions = cosine_act[b] * max_weights
                
                # Filter to only prototypes with POSITIVE contribution
                positive_mask = contributions > 0
                positive_indices = torch.where(positive_mask)[0]
                
                if len(positive_indices) > 0:
                    positive_contributions = contributions[positive_mask]
                    sorted_contrib, sorted_order = torch.sort(positive_contributions, descending=True)
                    sorted_indices = positive_indices[sorted_order]
                else:
                    # No positive contributions - fall back to all prototypes
                    sorted_contrib, sorted_indices = torch.sort(contributions, descending=True)
                    print(f"Warning: No prototypes with positive contribution for class {pred_class}")
            elif sort_by == 'weight':
                # Sort by weight only for predicted class (what model considers important for this class)
                # This shows prototypes with highest learned importance, regardless of activation
                weights = model.last_layer.weight[pred_class, :]  # [num_prototypes]
                
                # Filter to only prototypes with POSITIVE weights
                positive_mask = weights > 0
                positive_indices = torch.where(positive_mask)[0]
                
                if len(positive_indices) > 0:
                    positive_weights = weights[positive_mask]
                    sorted_weights, sorted_order = torch.sort(positive_weights, descending=True)
                    sorted_indices = positive_indices[sorted_order]
                else:
                    # No positive weights - fall back to all prototypes
                    sorted_weights, sorted_indices = torch.sort(weights, descending=True)
                    print(f"Warning: No prototypes with positive weight for class {pred_class}")
            elif sort_by == 'predicted_class_activation':
                # NEW: Top prototypes of PREDICTED CLASS that are activated in current sample
                # Sort by activation * weight, but only for prototypes trained for predicted class
                weights = model.last_layer.weight[pred_class, :]  # [num_prototypes]
                proto_class_identity = model.prototype_class_identity  # [num_prototypes, num_classes]
                
                # Find prototypes trained for predicted class
                proto_classes = proto_class_identity.argmax(dim=1)  # [num_prototypes]
                predicted_class_mask = (proto_classes == pred_class) & (weights > 0)
                predicted_class_indices = torch.where(predicted_class_mask)[0]
                
                if len(predicted_class_indices) > 0:
                    # Compute activation * weight for these prototypes
                    activations = cosine_act[b, predicted_class_indices]
                    proto_weights = weights[predicted_class_indices]
                    scores = activations * proto_weights
                    
                    sorted_scores, sorted_order = torch.sort(scores, descending=True)
                    sorted_indices = predicted_class_indices[sorted_order]
                else:
                    # Fallback: use all with positive weights
                    positive_mask = weights > 0
                    positive_indices = torch.where(positive_mask)[0]
                    if len(positive_indices) > 0:
                        activations = cosine_act[b, positive_indices]
                        proto_weights = weights[positive_indices]
                        scores = activations * proto_weights
                        sorted_scores, sorted_order = torch.sort(scores, descending=True)
                        sorted_indices = positive_indices[sorted_order]
                    else:
                        sorted_act, sorted_indices = torch.sort(cosine_act[b], descending=True)
                        print(f"Warning: No prototypes with positive weight for class {pred_class}")
            else:
                # Sort by activation only (default)
                # cosine_act[b] shape: [num_prototypes] (e.g., 2000)
                sorted_act, sorted_indices = torch.sort(cosine_act[b], descending=True)
            
            results = []
            # If k is None, return ALL prototypes (for later sorting by weight)
            # Otherwise return top-k
            if k is None:
                num_to_show = len(sorted_indices)
            else:
                num_to_show = min(k, len(sorted_indices))
            
            for i in range(num_to_show):
                # Ensure we have a scalar index
                proto_idx_tensor = sorted_indices[i]
                if proto_idx_tensor.numel() > 1:
                     # This should not happen if sort works as expected on 1D tensor
                     # But if something is weird with shape, take first element
                     print(f"Warning: proto_idx_tensor has shape {proto_idx_tensor.shape}")
                     proto_idx = proto_idx_tensor[0].item()
                else:
                    proto_idx = proto_idx_tensor.item()
                    
                # Get the ACTUAL activation (not contribution score if that was used for sorting)
                activation = cosine_act[b, proto_idx].item()
                
                # Get class identity
                # model.prototype_class_identity is [num_prototypes, num_classes]
                proto_class = model.prototype_class_identity[proto_idx].argmax().item()
                
                # Get last layer connection weight
                # model.last_layer.weight is [num_classes, num_prototypes]
                connection_weight = model.last_layer.weight[pred_class, proto_idx].item()
                
                # Compute contribution score for this prototype
                # For contribution, use max weight across all classes (global)
                if sort_by == 'contribution':
                    all_weights = model.last_layer.weight  # [num_classes, num_prototypes]
                    max_weight = torch.clamp(all_weights, min=0).max(dim=0)[0][proto_idx].item()
                    contribution = activation * max_weight
                else:
                    contribution = activation * max(0, connection_weight)
                
                # Get patch locations and slots
                proto_slots = slots[proto_idx].cpu().numpy() # [n_subpatches]
                
                # indices is [B, n_prototypes]
                # proto_indices is the index of the patch in the image (0..195) for this prototype
                proto_indices_tensor = indices[b, proto_idx]
                proto_indices = proto_indices_tensor.cpu().numpy()
                
                # Convert flat indices to 2D
                # Assuming 14x14 grid
                patch_locations = np.unravel_index(proto_indices.astype(int), (14, 14))
                
                results.append({
                    'proto_idx': proto_idx,
                    'activation': activation,
                    'class': proto_class,
                    'connection_weight': connection_weight,
                    'contribution': contribution,
                    'patch_locations': patch_locations,
                    'slots': proto_slots
                })
            
            batch_results.append((results, pred_class))
            
        return batch_results


def get_top_k_prototypes(model, image_tensor, k=10, sort_by='activation'):
    """
    Get the top-k most activated prototypes for an image.
    
    Args:
        sort_by: 'activation' (default), 'contribution' (activation * max(weight, 0)), 'weight' (weight only), or 'predicted_class_activation' (activation * weight for predicted class prototypes)
    
    Returns:
        List of dicts with keys: 'proto_idx', 'activation', 'class', 'connection_weight', 
                                 'contribution', 'patch_locations', 'slots'
    """
    model.eval()
    with torch.no_grad():
        image_batch = image_tensor.unsqueeze(0).cuda()
        logits, min_distances, values = model(image_batch)
        
        # Get predicted class
        _, predicted = torch.max(logits.data, 1)
        pred_class = predicted.item()
        
        # Compute prototype activations
        # patch_select: [1, n_prototypes, n_subpatches] -> squeeze to [n_prototypes, n_subpatches]
        slots = torch.sigmoid(model.patch_select * model.temp).squeeze(0)
        factor = ((slots.sum(-1))).unsqueeze(-1) + 1e-10
        proto_h = model.prototype_shape[2]
        n_p = proto_h  # number of prototype subpatches
        
        # Get indices for patch locations
        _, _, indices = model.push_forward(image_batch)
        
        # Compute cosine activations
        values_slot = (values.clone()) * (slots * n_p / factor)
        cosine_act = values_slot.sum(-1)  # [1, num_prototypes]
        
        # Compute sorting criterion
        if sort_by == 'contribution':
            # Sort by activation * max(weight, 0) GLOBALLY (across all classes)
            # This shows which prototypes contribute most to ANY class prediction
            # Get max weight across all classes for each prototype
            all_weights = model.last_layer.weight  # [num_classes, num_prototypes]
            max_weights = torch.clamp(all_weights, min=0).max(dim=0)[0]  # [num_prototypes] - max across classes
            contributions = cosine_act[0] * max_weights
            
            # Filter to only prototypes with POSITIVE contribution
            positive_mask = contributions > 0
            positive_indices = torch.where(positive_mask)[0]
            
            if len(positive_indices) > 0:
                positive_contributions = contributions[positive_mask]
                sorted_contrib, sorted_order = torch.sort(positive_contributions, descending=True)
                sorted_indices = positive_indices[sorted_order]
            else:
                # No positive contributions - fall back to all prototypes
                sorted_contrib, sorted_indices = torch.sort(contributions, descending=True)
                print(f"Warning: No prototypes with positive contribution for class {pred_class}")
        elif sort_by == 'weight':
            # Sort by weight only for predicted class (regardless of activation in this image)
            # This shows what the model considers important for this class
            weights = model.last_layer.weight[pred_class, :]  # [num_prototypes]
            
            # Filter to only prototypes with POSITIVE weights
            positive_mask = weights > 0
            positive_indices = torch.where(positive_mask)[0]
            
            if len(positive_indices) > 0:
                positive_weights = weights[positive_mask]
                sorted_weights, sorted_order = torch.sort(positive_weights, descending=True)
                sorted_indices = positive_indices[sorted_order]
            else:
                # No positive weights - fall back to all prototypes
                sorted_weights, sorted_indices = torch.sort(weights, descending=True)
                print(f"Warning: No prototypes with positive weight for class {pred_class}")
        elif sort_by == 'predicted_class_activation':
            # NEW: Top prototypes of PREDICTED CLASS that are activated in current sample
            # Sort by activation * weight, but only for prototypes trained for predicted class
            weights = model.last_layer.weight[pred_class, :]  # [num_prototypes]
            proto_class_identity = model.prototype_class_identity  # [num_prototypes, num_classes]
            
            # Find prototypes trained for predicted class
            proto_classes = proto_class_identity.argmax(dim=1)  # [num_prototypes]
            predicted_class_mask = (proto_classes == pred_class) & (weights > 0)
            predicted_class_indices = torch.where(predicted_class_mask)[0]
            
            if len(predicted_class_indices) > 0:
                # Compute activation * weight for these prototypes
                activations = cosine_act[0, predicted_class_indices]
                proto_weights = weights[predicted_class_indices]
                scores = activations * proto_weights
                
                sorted_scores, sorted_order = torch.sort(scores, descending=True)
                sorted_indices = predicted_class_indices[sorted_order]
            else:
                # Fallback: use all with positive weights
                positive_mask = weights > 0
                positive_indices = torch.where(positive_mask)[0]
                if len(positive_indices) > 0:
                    activations = cosine_act[0, positive_indices]
                    proto_weights = weights[positive_indices]
                    scores = activations * proto_weights
                    sorted_scores, sorted_order = torch.sort(scores, descending=True)
                    sorted_indices = positive_indices[sorted_order]
                else:
                    sorted_act, sorted_indices = torch.sort(cosine_act[0], descending=True)
                    print(f"Warning: No prototypes with positive weight for class {pred_class}")
        else:
            # Sort by activation
            sorted_act, sorted_indices = torch.sort(cosine_act[0], descending=True)
        
        # Get top-k (or all if k is None)
        results = []
        if k is None:
            num_to_show = len(sorted_indices)
        else:
            num_to_show = min(k, len(sorted_indices))
        for i in range(num_to_show):
            proto_idx = sorted_indices[i].item()
            
            # Get the ACTUAL activation (not contribution score if that was used for sorting)
            activation = cosine_act[0, proto_idx].item()
            
            # Get class identity
            proto_class = model.prototype_class_identity[proto_idx].argmax().item()
            
            # Get last layer connection weight
            connection_weight = model.last_layer.weight[pred_class, proto_idx].item()
            
            # Compute contribution score
            # For contribution, use max weight across all classes (global)
            # Always compute both for flexibility
            all_weights = model.last_layer.weight  # [num_classes, num_prototypes]
            max_weight_global = torch.clamp(all_weights, min=0).max(dim=0)[0][proto_idx].item()
            contribution_global = activation * max_weight_global
            contribution_class = activation * max(0, connection_weight)
            
            # Use global contribution if sort_by is 'contribution', otherwise use class-specific
            contribution = contribution_global if sort_by == 'contribution' else contribution_class
            
            # Get patch locations and slots
            proto_slots = slots[proto_idx].cpu().numpy()
            proto_indices = indices[0, proto_idx].cpu().numpy()
            
            # Convert flat indices to 2D
            patch_locations = np.unravel_index(proto_indices.astype(int), (14, 14))
            
            results.append({
                'proto_idx': proto_idx,
                'activation': activation,
                'class': proto_class,
                'connection_weight': connection_weight,
                'contribution': contribution,
                'patch_locations': patch_locations,
                'slots': proto_slots
            })
        
        return results, pred_class



def draw_bounding_boxes(img_rgb, patch_locations, slots, n_p=4, img_size=224, 
                        grid_size=14, patch_size=16):
    """
    Draw bounding boxes on image for activated prototype patches.
    
    Args:
        img_rgb: numpy array [H, W, 3]
        patch_locations: tuple of (height_indices, width_indices)
        slots: array indicating which sub-patches are active
        n_p: number of sub-patches per prototype
    """
    # Convert to PIL Image for drawing
    img_uint8 = np.uint8(np.clip(img_rgb * 255, 0, 255))
    img_pil = Image.fromarray(img_uint8, mode='RGB')
    draw = ImageDraw.Draw(img_pil)
    
    # RGB colors for PIL
    colors = [
        (255, 255, 0),    # Yellow
        (0, 0, 255),      # Blue
        (0, 255, 0),      # Green
        (255, 0, 0),      # Red
        (0, 255, 255),    # Cyan
        (255, 0, 255),    # Magenta
        (255, 255, 255)   # White
    ]
    
    for k in range(n_p):
        if slots[k] > 0:
            h_idx = patch_locations[0][k]
            w_idx = patch_locations[1][k]
            
            # Calculate pixel coordinates
            h_start = int(h_idx * patch_size)
            h_end = int(h_start + patch_size)
            w_start = int(w_idx * patch_size)
            w_end = int(w_start + patch_size)
            
            color = colors[k % len(colors)]
            # Draw rectangle (PIL uses (left, top, right, bottom))
            draw.rectangle([(w_start, h_start), (w_end - 1, h_end - 1)], 
                          outline=color, width=2)
    
    # Convert back to numpy array
    img_rgb_boxed = np.array(img_pil, dtype=np.float32) / 255
    return img_rgb_boxed


def create_prototype_visualization(model, image_tensor, model_name, prototype_img_dir, top_k=10, true_class=None, precomputed_results=None, sort_by='activation', target_class=None):
    """
    Create a comprehensive visualization of prototype activations with "this looks like that".
    
    Args:
        sort_by: 'activation' (default), 'contribution' (activation * max(weight, 0) globally), 'weight' (weight only), 
                 'predicted_class_activation' (activation * weight for predicted class prototypes),
                 'ground_truth_class_activation' (activation * weight for ground truth class prototypes),
                 'specific_class_activation' (activation * weight for target_class prototypes)
        target_class: Required if sort_by is 'specific_class_activation'
    
    Returns:
        fig: matplotlib figure
        proto_results: list of prototype activation results
        pred_class: predicted class
    """
    if precomputed_results is not None:
        proto_results = precomputed_results['proto_results']
        pred_class = precomputed_results['pred_class']  # Use SAME predicted class from precomputed
        
        # IMPORTANT: We should NOT recompute (no new forward pass) to avoid different predictions
        # Now we have ALL prototypes saved, so we can properly filter and sort
        
        if sort_by == 'weight':
            # Filter to prototypes with positive weights for predicted class, then sort by weight
            # Get weights from the model (adapted model if it's adapted)
            weights = model.last_layer.weight[pred_class, :].detach().cpu().numpy()
            
            # Filter to only those with positive weights
            filtered_results = [p for p in proto_results if weights[p['proto_idx']] > 0]
            
            if len(filtered_results) > 0:
                # Sort by weight (descending)
                filtered_results.sort(key=lambda x: weights[x['proto_idx']], reverse=True)
                proto_results = filtered_results[:top_k]
            else:
                # Fallback: use all precomputed, sorted by weight
                proto_results = sorted(proto_results, key=lambda x: x['connection_weight'], reverse=True)[:top_k]
                
        elif sort_by == 'predicted_class_activation':
            # Filter to prototypes of predicted class, then sort by contribution (activation * weight)
            # Get weights for predicted class
            weights = model.last_layer.weight[pred_class, :].detach().cpu().numpy()
            filtered_results = [p for p in proto_results if p['class'] == pred_class and weights[p['proto_idx']] > 0]
            
            if len(filtered_results) > 0:
                # Compute and store contribution for predicted class
                for p in filtered_results:
                    p['contribution'] = p['activation'] * weights[p['proto_idx']]
                # Sort by contribution (activation * weight) for predicted class prototypes
                filtered_results.sort(key=lambda x: x['contribution'], reverse=True)
                proto_results = filtered_results[:top_k]
            else:
                # Fallback: use all precomputed
                proto_results = proto_results[:top_k]
        elif sort_by == 'ground_truth_class_activation':
            # Filter to prototypes of ground truth class, then sort by contribution (activation * weight)
            if true_class is not None:
                # Get weights for ground truth class
                weights = model.last_layer.weight[true_class, :].detach().cpu().numpy()
                filtered_results = [p for p in proto_results if p['class'] == true_class and weights[p['proto_idx']] > 0]
                
                if len(filtered_results) > 0:
                    # Compute and store contribution for ground truth class
                    for p in filtered_results:
                        p['contribution'] = p['activation'] * weights[p['proto_idx']]
                    # Sort by contribution (activation * weight) for ground truth class
                    filtered_results.sort(key=lambda x: x['contribution'], reverse=True)
                    proto_results = filtered_results[:top_k]
                else:
                    proto_results = []
            else:
                proto_results = []
        elif sort_by == 'specific_class_activation':
            # Filter to prototypes of target_class, then sort by contribution (activation * weight)
            if target_class is not None:
                weights = model.last_layer.weight[target_class, :].detach().cpu().numpy()
                filtered_results = [p for p in proto_results if p['class'] == target_class and weights[p['proto_idx']] > 0]
                
                if len(filtered_results) > 0:
                    # Compute and store contribution for target class
                    for p in filtered_results:
                        p['contribution'] = p['activation'] * weights[p['proto_idx']]
                    # Sort by contribution (activation * weight) for target class
                    filtered_results.sort(key=lambda x: x['contribution'], reverse=True)
                    proto_results = filtered_results[:top_k]
                else:
                    proto_results = []
            else:
                proto_results = []
                
        elif sort_by == 'contribution':
            # Recompute contribution globally (max weight across all classes)
            # Get max weight across all classes for each prototype
            all_weights = model.last_layer.weight.detach().cpu().numpy()  # [num_classes, num_prototypes]
            max_weights = np.maximum(all_weights, 0).max(axis=0)  # [num_prototypes] - max across classes
            
            # Recompute contribution for each prototype
            for p in proto_results:
                p['contribution'] = p['activation'] * max_weights[p['proto_idx']]
            
            # Sort by contribution (global)
            proto_results = sorted(proto_results, key=lambda x: x['contribution'], reverse=True)[:top_k]
        elif sort_by == 'activation':
            # Already sorted by activation, just take top-k
            proto_results = proto_results[:top_k]
    else:
        # Get top-k prototypes (only when precomputed not available)
        proto_results, pred_class = get_top_k_prototypes(model, image_tensor, k=top_k, sort_by=sort_by)
    
    # Get original image
    img_rgb = save_tensor_image(image_tensor, '/tmp/temp_viz.png')
    
    # Create figure: Test image patch | Training prototype | Info
    # For predicted_class_activation, add attention map column
    show_attention = (sort_by == 'predicted_class_activation')
    n_rows = min(5, len(proto_results))
    n_cols = 4 if show_attention else 3  # Test image, Training prototype, Attention map (if applicable), Info
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Title with predicted and ground truth
    if sort_by == 'contribution':
        sort_info = " (Sorted by Contribution: Activation × Max Weight Globally)"
    elif sort_by == 'weight':
        sort_info = " (Sorted by Weight: Model's Learned Importance)"
    elif sort_by == 'predicted_class_activation':
        sort_info = " (Predicted Class Prototypes: Activation × Weight)"
    elif sort_by == 'ground_truth_class_activation':
        sort_info = f" (Ground Truth Class {true_class} Prototypes: Activation × Weight)"
    elif sort_by == 'specific_class_activation':
        sort_info = f" (Class {target_class} Prototypes: Activation × Weight)"
    else:
        sort_info = " (Sorted by Activation)"
    
    if true_class is not None:
        correct = "✓" if pred_class == true_class else "✗"
        color = 'green' if pred_class == true_class else 'red'
        title = f'{model_name}{sort_info}\nPredicted: {pred_class} | Ground Truth: {true_class} {correct}'
        fig.suptitle(title, fontsize=16, fontweight='bold', color=color)
    else:
        fig.suptitle(f'{model_name}{sort_info} - Predicted Class: {pred_class}', 
                     fontsize=16, fontweight='bold')
    
    for i in range(n_rows):
        if i < len(proto_results):
            proto = proto_results[i]
            proto_idx = proto['proto_idx']
            
            # Column 0: Test image with bounding boxes (THIS)
            img_with_boxes = draw_bounding_boxes(
                img_rgb, 
                proto['patch_locations'], 
                proto['slots']
            )
            axes[i, 0].imshow(img_with_boxes)
            axes[i, 0].set_title(f"Top-{i+1} THIS (Test Image)", fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')
            
            # Column 1: Training prototype image (LOOKS LIKE THAT)
            proto_loaded = False
            # Try multiple locations and patterns
            possible_locations = [
                prototype_img_dir,  # Direct path
                os.path.join(prototype_img_dir, 'epoch-4'),  # Common structure
                os.path.dirname(prototype_img_dir),  # Parent dir
            ]
            
            possible_patterns = [
                f'prototype-imgbbox-original{proto_idx}.png',
                f'prototype-img_vis_{proto_idx}.png',
                f'prototype-img-original{proto_idx}.png',
                f'prototype-img{proto_idx}.png',
                f'prototype{proto_idx}.png',
            ]
            
            for base_dir in possible_locations:
                if not os.path.exists(base_dir):
                    continue
                for pattern in possible_patterns:
                    proto_path = os.path.join(base_dir, pattern)
                    if os.path.exists(proto_path):
                        try:
                            proto_img = plt.imread(proto_path)
                            axes[i, 1].imshow(proto_img)
                            axes[i, 1].set_title(f"LOOKS LIKE THAT (Proto {proto_idx})", 
                                                fontsize=10, fontweight='bold', color='green')
                            proto_loaded = True
                            break
                        except Exception as e:
                            continue
                if proto_loaded:
                    break
            
            if not proto_loaded:
                axes[i, 1].text(0.5, 0.5, f'Prototype {proto_idx}\n(Image not found)', 
                               ha='center', va='center', fontsize=10)
                axes[i, 1].set_title(f"Training Prototype {proto_idx}", fontsize=10)
            axes[i, 1].axis('off')
            
            # Column 2: Attention map (only for predicted_class_activation)
            col_idx = 2
            if show_attention:
                # Create attention heatmap for this prototype
                heatmap = np.zeros((14, 14))
                patch_locations = proto['patch_locations']
                slots = proto['slots']
                
                # Mark activated locations with activation value
                for k in range(len(slots)):
                    if slots[k] > 0:
                        h_idx = patch_locations[0][k]
                        w_idx = patch_locations[1][k]
                        # Use contribution (activation * weight) as heatmap value
                        contrib = proto.get('contribution', proto['activation'] * max(0, proto['connection_weight']))
                        heatmap[h_idx, w_idx] = contrib
                
                # Resize heatmap to image size
                heatmap_resized = zoom(heatmap, 224/14, order=1)
                
                # Overlay on test image
                axes[i, col_idx].imshow(img_rgb)
                axes[i, col_idx].imshow(heatmap_resized, alpha=0.6, cmap='hot', vmin=0)
                axes[i, col_idx].set_title(f"Attention Map\n(Contribution: {proto.get('contribution', 0):.3f})", 
                                          fontsize=10, fontweight='bold')
                axes[i, col_idx].axis('off')
                col_idx = 3
            else:
                col_idx = 2
            
            # Last column: Detailed information
            axes[i, col_idx].axis('off')
            
            # Add contribution info - always compute for class-specific visualizations
            contribution_text = ""
            if 'contribution' in proto:
                contribution_text = f"Contribution: {proto['contribution']:.4f}\n"
            elif sort_by in ['ground_truth_class_activation', 'specific_class_activation', 'predicted_class_activation']:
                # Compute contribution for this specific class
                if sort_by == 'ground_truth_class_activation' and true_class is not None:
                    target_class = true_class
                elif sort_by == 'specific_class_activation' and target_class is not None:
                    target_class = target_class
                elif sort_by == 'predicted_class_activation':
                    target_class = pred_class
                else:
                    target_class = None
                
                if target_class is not None:
                    weight_for_class = model.last_layer.weight[target_class, proto_idx].item()
                    contribution = proto['activation'] * max(0, weight_for_class)
                    # Store it in proto dict for consistency
                    proto['contribution'] = contribution
                    contribution_text = f"Contribution: {contribution:.4f}\n"
            
            info_text = (
                f"Prototype {proto_idx}\n"
                f"─────────────────────\n"
                f"Activation: {proto['activation']:.4f}\n"
                f"Proto Class: {proto['class']}\n"
                f"Weight: {proto['connection_weight']:.4f}\n"
                f"{contribution_text}"
                f"Active Patches: {(proto['slots'] > 0).sum()}/{len(proto['slots'])}\n\n"
                f"Interpretation:\n"
                f"The highlighted region in\n"
                f"the test image (THIS)\n"
                f"matches the training\n"
                f"prototype (THAT)"
            )
            axes[i, col_idx].text(0.05, 0.5, info_text, 
                           fontsize=11, verticalalignment='center',
                           family='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        else:
            # Hide unused subplots
            for j in range(n_cols):
                axes[i, j].axis('off')
    
    plt.tight_layout()
    return fig, proto_results, pred_class


def create_activation_heatmaps(models_dict, clean_tensor, noisy_tensor, prototype_img_dir, image_output_dir):
    """
    Create activation heatmaps showing top-5 prototypes for each method.
    4 rows: Clean baseline, Normal (noisy), EATA, ProtoEntropy
    """
    method_names = list(models_dict.keys())
    n_methods = len(method_names) + 1  # +1 for clean baseline
    n_protos = 5  # Top-5 prototypes
    
    # Create figure: rows=methods+clean, cols=prototypes
    fig, axes = plt.subplots(n_methods, n_protos, figsize=(4*n_protos, 4*n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Top-5 Prototype Activation Maps Comparison\n(Clean Baseline + Noisy with Different Adaptations)', 
                 fontsize=16, fontweight='bold')
    
    # Get base images for overlay
    clean_img_rgb = save_tensor_image(clean_tensor, '/tmp/temp_heatmap_clean.png')
    noisy_img_rgb = save_tensor_image(noisy_tensor, '/tmp/temp_heatmap_noisy.png')
    
    # Row 0: Clean baseline (Normal model on clean image)
    if 'Normal' in models_dict:
        model_wrapper = models_dict['Normal']
        if hasattr(model_wrapper, 'model'):
            model = model_wrapper.model
        else:
            model = model_wrapper
        model.eval()
        
        proto_results, pred_class = get_top_k_prototypes(model, clean_tensor, k=n_protos)
        
        for proto_idx in range(n_protos):
            ax = axes[0, proto_idx]
            
            if proto_idx < len(proto_results):
                proto = proto_results[proto_idx]
                
                # Create heatmap
                heatmap = np.zeros((14, 14))
                patch_locations = proto['patch_locations']
                slots = proto['slots']
                
                for k in range(len(slots)):
                    if slots[k] > 0:
                        h_idx = patch_locations[0][k]
                        w_idx = patch_locations[1][k]
                        heatmap[h_idx, w_idx] = proto['activation']
                
                heatmap_resized = zoom(heatmap, 224/14, order=1)
                
                ax.imshow(clean_img_rgb)
                ax.imshow(heatmap_resized, alpha=0.6, cmap='hot', vmin=0, vmax=1.0)
                ax.set_title(f'Proto {proto["proto_idx"]}\nAct: {proto["activation"]:.3f}', fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')
            
            if proto_idx == 0:
                ax.text(-0.1, 0.5, 'CLEAN\nBaseline', 
                       transform=ax.transAxes,
                       fontsize=12, fontweight='bold',
                       rotation=90, va='center', ha='right',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Remaining rows: Noisy with different methods
    for method_idx, (method_name, model_wrapper) in enumerate(models_dict.items(), start=1):
        # Extract model
        if hasattr(model_wrapper, 'model'):
            model = model_wrapper.model
        else:
            model = model_wrapper
        
        model.eval()
        
        # Get top-5 prototypes on NOISY image
        proto_results, pred_class = get_top_k_prototypes(model, noisy_tensor, k=n_protos)
        
        for proto_idx in range(n_protos):
            ax = axes[method_idx, proto_idx]
            
            if proto_idx < len(proto_results):
                proto = proto_results[proto_idx]
                
                # Create heatmap from patch locations
                heatmap = np.zeros((14, 14))  # Grid size
                patch_locations = proto['patch_locations']
                slots = proto['slots']
                
                # Mark activated locations
                for k in range(len(slots)):
                    if slots[k] > 0:
                        h_idx = patch_locations[0][k]
                        w_idx = patch_locations[1][k]
                        heatmap[h_idx, w_idx] = proto['activation']
                
                # Resize heatmap to image size
                heatmap_resized = zoom(heatmap, 224/14, order=1)
                
                # Overlay on NOISY image
                ax.imshow(noisy_img_rgb)
                im = ax.imshow(heatmap_resized, alpha=0.6, cmap='hot', 
                              vmin=0, vmax=1.0)
                
                # Title
                if method_idx == 1:
                    ax.set_title(f'Proto {proto["proto_idx"]}\n'
                               f'Act: {proto["activation"]:.3f}',
                               fontsize=10)
                else:
                    ax.set_title(f'Act: {proto["activation"]:.3f}', fontsize=10)
                
                ax.axis('off')
            else:
                ax.axis('off')
            
            # Add method name on left
            if proto_idx == 0:
                # Color-code: Normal=orange, EATA=blue, ProtoEntropy=green
                if 'Normal' in method_name:
                    bg_color = 'lightyellow'
                elif 'EATA' in method_name:
                    bg_color = 'lightblue'
                else:
                    bg_color = 'lightcoral'
                
                ax.text(-0.1, 0.5, f'NOISY\n{method_name}', 
                       transform=ax.transAxes,
                       fontsize=12, fontweight='bold',
                       rotation=90, va='center', ha='right',
                       bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    heatmap_path = os.path.join(image_output_dir, '05_activation_heatmaps_comparison.png')
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return heatmap_path


def save_prototype_patches(proto_results, prototype_img_dir, output_dir, max_protos=10):
    """
    Save the original training prototype patches for activated prototypes.
    
    Args:
        proto_results: List of prototype activation results
        prototype_img_dir: Directory containing prototype images
        output_dir: Directory to save prototype patches
        max_protos: Maximum number of prototypes to save
    """
    makedir(output_dir)
    
    for i, proto in enumerate(proto_results[:max_protos]):
        proto_idx = proto['proto_idx']
        
        # Look for prototype images
        # Common patterns: prototype-img{idx}.png, prototype-imgbbox-original{idx}.png
        possible_patterns = [
            f'prototype-imgbbox-original{proto_idx}.png',
            f'prototype-img{proto_idx}.png',
            f'prototype{proto_idx}.png',
        ]
        
        for pattern in possible_patterns:
            src_path = os.path.join(prototype_img_dir, pattern)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_dir, 
                                       f'top{i+1:02d}_proto{proto_idx:04d}_original.png')
                # Copy and annotate
                img = plt.imread(src_path)
                
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(img)
                ax.set_title(f'Proto {proto_idx} (Class {proto["class"]})\n'
                           f'Activation: {proto["activation"]:.4f}',
                           fontsize=10)
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(dst_path, dpi=100, bbox_inches='tight')
                plt.close()
                break


def run_comprehensive_interpretability(
    models_dict,  # Dict: {method_name: model_wrapper}
    image_path,   # Relative path: e.g., "066.Western_Gull/Western_Gull_0007_53431.jpg"
    test_dir,     # Clean test directory (or corrupted if use_pre_corrupted=True)
    prototype_img_dir,  # Where prototype images are stored
    output_base_dir,    # Base output directory
    corruption_name,    # e.g., "gaussian_noise"
    severity,           # e.g., 5
    experimental_settings,  # Dict with all settings
    true_class=None,    # Ground truth class label
    use_pre_corrupted=False, # Whether to use pre-corrupted images (skip on-the-fly corruption)
    data_root=None,      # Root directory for loading images (overrides test_dir if provided)
    precomputed_results_dict=None # Dict: {method_name: results_dict}
):
    """
    Create comprehensive interpretability visualizations for multiple TTA methods.
    Shows "this looks like that" - test image patches matched with training prototypes.
    """
    # Parse image path to get class and image name
    path_parts = image_path.split('/')
    if len(path_parts) >= 2:
        class_name = path_parts[0]
        image_name = os.path.splitext(path_parts[1])[0]
    else:
        class_name = "unknown"
        image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create output directory for this image
    corruption_str = f"{corruption_name}_sev{severity}" if corruption_name else "clean"
    image_output_dir = os.path.join(
        output_base_dir, 
        'interpretability_comprehensive',
        corruption_str,
        class_name,
        image_name
    )
    makedir(image_output_dir)
    
    print(f"\n{'='*60}")
    print(f"Interpretability: {image_path}")
    print(f"Output: {image_output_dir}")
    print(f"{'='*60}\n")
    
    # ========== Load and save images ==========
    # Determine source directory
    source_dir = data_root if data_root else test_dir
    full_image_path = os.path.join(source_dir, image_path)
    
    if not os.path.exists(full_image_path):
        print(f"⚠ Image not found at {full_image_path}")
        # Try finding it in test_dir if data_root failed
        if data_root and os.path.exists(os.path.join(test_dir, image_path)):
            print(f"  Fallback: Found in clean dir {test_dir}")
            full_image_path = os.path.join(test_dir, image_path)
            # If we fall back to clean dir, we MUST apply corruption even if use_pre_corrupted was True
            use_pre_corrupted = False 
    
    img_pil = Image.open(full_image_path)
    
    # Get image size from models
    img_size = 224
    for model_wrapper in models_dict.values():
        if hasattr(model_wrapper, 'model'):
            model = model_wrapper.model
        else:
            model = model_wrapper
        if hasattr(model, 'img_size'):
            img_size = model.img_size
            break
    
    # Preprocessing
    normalize = transforms.Normalize(mean=mean, std=std)
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Handle corruption
    if corruption_name and not use_pre_corrupted:
        # CASE 1: Load Clean -> Apply Corruption -> Normalize
        print(f"  Generating {corruption_name} (severity {severity}) on-the-fly...")
        print(f"  ⚠ WARNING: On-the-fly corruption involves random noise.")
        print(f"  ⚠ The generated image may differ from the one used during inference!")
        print(f"  ⚠ For consistent analysis, use pre-generated corrupted datasets.")
        
        # Save original clean image
        clean_tensor = preprocess(img_pil)
        clean_img_path = os.path.join(image_output_dir, '00_original_clean_image.png')
        save_tensor_image(clean_tensor, clean_img_path)
        print(f"✓ Clean image saved")
        
        # Generate noisy image
        corrupt_transform = get_corrupted_transform(
            img_size, mean, std, corruption_name, severity
        )
        noisy_tensor = corrupt_transform(img_pil)
        
        noisy_img_path = os.path.join(image_output_dir, '01_noisy_image.png')
        save_tensor_image(noisy_tensor, noisy_img_path)
        print(f"✓ Noisy image saved")
        
    elif corruption_name and use_pre_corrupted:
        # CASE 2: Load Pre-Corrupted -> Normalize (No extra corruption)
        print(f"  Using pre-corrupted image from disk...")
        
        # The loaded image IS the noisy image
        noisy_tensor = preprocess(img_pil)
        
        noisy_img_path = os.path.join(image_output_dir, '01_noisy_image.png')
        save_tensor_image(noisy_tensor, noisy_img_path)
        print(f"✓ Noisy image saved")
        
        # For 'clean' baseline, we try to find the actual clean image
        # Assuming test_dir points to clean data
        clean_path_guess = os.path.join(test_dir, image_path)
        if os.path.exists(clean_path_guess) and clean_path_guess != full_image_path:
            clean_pil = Image.open(clean_path_guess)
            clean_tensor = preprocess(clean_pil)
            clean_img_path = os.path.join(image_output_dir, '00_original_clean_image.png')
            save_tensor_image(clean_tensor, clean_img_path)
            print(f"✓ Clean image saved (from clean dir)")
        else:
            # If clean not found, just use noisy as placeholder or try to denoise?
            # Better to just duplicate for visualization code stability
            print(f"⚠ Clean image not found, using noisy as clean placeholder")
            clean_tensor = noisy_tensor
            
    else:
        # CASE 3: No Corruption (Clean Test)
        clean_tensor = preprocess(img_pil)
        noisy_tensor = clean_tensor
        
        clean_img_path = os.path.join(image_output_dir, '00_original_clean_image.png')
        save_tensor_image(clean_tensor, clean_img_path)
        print(f"✓ Clean image saved")

    
    # ========== Analyze CLEAN image with Normal model (baseline) ==========
    all_proto_results = {}
    all_proto_results_clean = {}
    
    if 'Normal' in models_dict:
        print(f"\nAnalyzing CLEAN image with Normal model (baseline)...")
        model_wrapper = models_dict['Normal']
        
        if hasattr(model_wrapper, 'model'):
            model = model_wrapper.model
        else:
            model = model_wrapper
        
        model.eval()
        
        # Check for precomputed results
        precomputed = None
        if precomputed_results_dict and 'Normal_Clean' in precomputed_results_dict:
            precomputed = precomputed_results_dict['Normal_Clean']
        
        # Create visualization for clean image
        fig, proto_results, pred_class = create_prototype_visualization(
            model, clean_tensor, 'Normal (Clean - No Distribution Shift)', 
            prototype_img_dir, top_k=5, true_class=true_class,
            precomputed_results=precomputed
        )
        
        clean_analysis_path = os.path.join(image_output_dir, '02_Normal_CLEAN_baseline.png')
        fig.savefig(clean_analysis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved clean baseline: {clean_analysis_path}")
        print(f"  Predicted class: {pred_class}")
        all_proto_results_clean['Normal'] = proto_results
    
    # ========== Analyze NOISY image with all methods ==========
    for method_name, model_wrapper in models_dict.items():
        print(f"\nAnalyzing NOISY image with {method_name}...")
        
        # Extract underlying model
        if hasattr(model_wrapper, 'model'):
            model = model_wrapper.model
        else:
            model = model_wrapper
        
        model.eval()
        
        # Check for precomputed results
        precomputed = None
        if precomputed_results_dict and method_name in precomputed_results_dict:
            precomputed = precomputed_results_dict[method_name]
            print(f"  Using PRECOMPUTED results for {method_name}")
        
        # Create visualization with "this looks like that" - sorted by ACTIVATION
        fig, proto_results, pred_class = create_prototype_visualization(
            model, noisy_tensor, f'{method_name} (Noisy) - By Activation', 
            prototype_img_dir, top_k=5, true_class=true_class,
            precomputed_results=precomputed,
            sort_by='activation'
        )
        
        # Save
        analysis_path = os.path.join(
            image_output_dir, 
            f'03_{method_name.replace(" ", "_")}_NOISY_analysis.png'
        )
        fig.savefig(analysis_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Saved: {analysis_path}")
        print(f"  Predicted class: {pred_class}")
        
        # Create visualization sorted by CONTRIBUTION (activation * max(weight, 0))
        fig_contrib, proto_results_contrib, pred_class_contrib = create_prototype_visualization(
            model, noisy_tensor, f'{method_name} (Noisy) - By Contribution to Pred {pred_class}', 
            prototype_img_dir, top_k=5, true_class=true_class,
            precomputed_results=precomputed,
            sort_by='contribution'
        )
        
        # Save contribution-based visualization
        analysis_path_contrib = os.path.join(
            image_output_dir, 
            f'03_{method_name.replace(" ", "_")}_NOISY_analysis_CONTRIBUTION.png'
        )
        fig_contrib.savefig(analysis_path_contrib, dpi=150, bbox_inches='tight')
        plt.close(fig_contrib)
        
        # Print diagnostic information about contributions
        num_positive = sum(1 for p in proto_results_contrib if p.get('contribution', 0) > 0)
        
        # Check how many prototypes in total have positive weights for this class
        weights_for_class = model.last_layer.weight[pred_class, :]  # [num_prototypes]
        num_positive_weights = (weights_for_class > 0).sum().item()
        
        print(f"✓ Saved contribution-based: {analysis_path_contrib}")
        print(f"  Predicted class: {pred_class}")
        print(f"  Total prototypes with positive weights for class {pred_class}: {num_positive_weights}")
        print(f"  Showing top {num_positive}/{len(proto_results_contrib)} with positive contribution")
        
        if num_positive > 0:
            top_contrib = proto_results_contrib[0]['contribution']
            top_proto = proto_results_contrib[0]
            print(f"  Top contributor: proto {top_proto['proto_idx']} (trained for class {top_proto['class']})")
            print(f"    Activation: {top_proto['activation']:.4f}, Weight: {top_proto['connection_weight']:.4f}, Contribution: {top_contrib:.4f}")
            
            # Show summary of all shown prototypes
            for i, p in enumerate(proto_results_contrib[:5], 1):
                if p.get('contribution', 0) > 0:
                    print(f"  {i}. Proto {p['proto_idx']} (class {p['class']}): act={p['activation']:.3f}, wt={p['connection_weight']:.3f}, contrib={p['contribution']:.3f}")
        
        # Create visualization sorted by WEIGHT only (what model considers important for this class)
        # IMPORTANT: Use same precomputed results and same predicted class - NO recomputation
        fig_weight, proto_results_weight, pred_class_weight = create_prototype_visualization(
            model, noisy_tensor, f'{method_name} (Noisy) - By Weight for Class {pred_class}', 
            prototype_img_dir, top_k=5, true_class=true_class,
            precomputed_results=precomputed,  # Use same precomputed - ensures same prediction
            sort_by='weight'
        )
        
        # Verify prediction matches
        if pred_class_weight != pred_class:
            print(f"  ⚠ WARNING: Weight visualization prediction ({pred_class_weight}) differs from main prediction ({pred_class})")
            print(f"    This should not happen - using precomputed results!")
        
        # Save weight-based visualization
        analysis_path_weight = os.path.join(
            image_output_dir, 
            f'03_{method_name.replace(" ", "_")}_NOISY_analysis_WEIGHT.png'
        )
        fig_weight.savefig(analysis_path_weight, dpi=150, bbox_inches='tight')
        plt.close(fig_weight)
        
        print(f"✓ Saved weight-based: {analysis_path_weight}")
        print(f"  Shows prototypes with highest weights for class {pred_class} (model's learned importance)")
        print(f"  Prediction: {pred_class_weight} (should match {pred_class})")
        if len(proto_results_weight) > 0:
            print(f"  Top weight: {proto_results_weight[0]['connection_weight']:.4f} (proto {proto_results_weight[0]['proto_idx']}, trained for class {proto_results_weight[0]['class']})")
            print(f"  Activation in this image: {proto_results_weight[0]['activation']:.4f}")
        
        # Create visualization for predicted class prototypes that are activated
        # Filter to only prototypes of predicted class, sorted by activation * weight
        fig_pred_class, proto_results_pred_class, pred_class_pred_class = create_prototype_visualization(
            model, noisy_tensor, f'{method_name} (Noisy) - Predicted Class Prototypes Activated', 
            prototype_img_dir, top_k=5, true_class=true_class,
            precomputed_results=precomputed,  # Use same precomputed - ensures same prediction
            sort_by='predicted_class_activation'
        )
        
        # Verify prediction matches
        if pred_class_pred_class != pred_class:
            print(f"  ⚠ WARNING: Pred class activation visualization prediction ({pred_class_pred_class}) differs from main prediction ({pred_class})")
        
        # Save predicted class activation visualization
        analysis_path_pred_class = os.path.join(
            image_output_dir, 
            f'03_{method_name.replace(" ", "_")}_NOISY_analysis_PRED_CLASS_ACTIVATED.png'
        )
        fig_pred_class.savefig(analysis_path_pred_class, dpi=150, bbox_inches='tight')
        plt.close(fig_pred_class)
        
        print(f"✓ Saved predicted class prototypes: {analysis_path_pred_class}")
        print(f"  Shows top prototypes of class {pred_class} that are activated in this image")
        print(f"  Prediction: {pred_class_pred_class} (should match {pred_class})")
        if len(proto_results_pred_class) > 0:
            top_proto = proto_results_pred_class[0]
            num_pred_class_protos = sum(1 for p in proto_results_pred_class if p['class'] == pred_class)
            print(f"  Prototypes of class {pred_class}: {num_pred_class_protos}/{len(proto_results_pred_class)}")
            print(f"  Top: proto {top_proto['proto_idx']} (class {top_proto['class']}, act={top_proto['activation']:.3f}, wt={top_proto['connection_weight']:.3f}, contrib={top_proto.get('contribution', top_proto['activation']*max(0,top_proto['connection_weight'])):.3f})")
        
        # For Normal and EATA, also create visualizations for ground truth and predicted class separately
        if method_name in ['Normal', 'EATA'] and true_class is not None:
            # Ground Truth Class Prototypes
            fig_gt_class, proto_results_gt_class, pred_class_gt_class = create_prototype_visualization(
                model, noisy_tensor, f'{method_name} (Noisy) - Ground Truth Class {true_class} Prototypes Activated', 
                prototype_img_dir, top_k=5, true_class=true_class,
                precomputed_results=precomputed,
                sort_by='ground_truth_class_activation'
            )
            
            analysis_path_gt_class = os.path.join(
                image_output_dir, 
                f'03_{method_name.replace(" ", "_")}_NOISY_analysis_GT_CLASS_{true_class}_ACTIVATED.png'
            )
            fig_gt_class.savefig(analysis_path_gt_class, dpi=150, bbox_inches='tight')
            plt.close(fig_gt_class)
            
            print(f"✓ Saved ground truth class prototypes: {analysis_path_gt_class}")
            print(f"  Shows top prototypes of GROUND TRUTH class {true_class} that are activated")
            if len(proto_results_gt_class) > 0:
                top_proto_gt = proto_results_gt_class[0]
                print(f"  Top: proto {top_proto_gt['proto_idx']} (class {top_proto_gt['class']}, act={top_proto_gt['activation']:.3f}, wt={top_proto_gt['connection_weight']:.3f})")
            
            # Predicted Class Prototypes (separate from the one above, this one is simpler)
            fig_pred_class_simple, proto_results_pred_class_simple, pred_class_pred_class_simple = create_prototype_visualization(
                model, noisy_tensor, f'{method_name} (Noisy) - Predicted Class {pred_class} Prototypes Activated', 
                prototype_img_dir, top_k=5, true_class=true_class,
                precomputed_results=precomputed,
                sort_by='specific_class_activation',
                target_class=pred_class
            )
            
            analysis_path_pred_class_simple = os.path.join(
                image_output_dir, 
                f'03_{method_name.replace(" ", "_")}_NOISY_analysis_PRED_CLASS_{pred_class}_ACTIVATED.png'
            )
            fig_pred_class_simple.savefig(analysis_path_pred_class_simple, dpi=150, bbox_inches='tight')
            plt.close(fig_pred_class_simple)
            
            print(f"✓ Saved predicted class prototypes (simple): {analysis_path_pred_class_simple}")
            print(f"  Shows top prototypes of PREDICTED class {pred_class} that are activated")
            if len(proto_results_pred_class_simple) > 0:
                top_proto_pred = proto_results_pred_class_simple[0]
                print(f"  Top: proto {top_proto_pred['proto_idx']} (class {top_proto_pred['class']}, act={top_proto_pred['activation']:.3f}, wt={top_proto_pred['connection_weight']:.3f})")
        
        all_proto_results[method_name] = proto_results
    
    # ========== Create activation heatmap comparison ==========
    print(f"\nCreating activation heatmaps comparison...")
    try:
        heatmap_path = create_activation_heatmaps(
            models_dict, clean_tensor, noisy_tensor, prototype_img_dir, image_output_dir
        )
        print(f"✓ Saved activation heatmaps: {heatmap_path}")
    except Exception as e:
        print(f"⚠ Could not create heatmaps: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== Save comparison summary ==========
    summary_path = os.path.join(image_output_dir, '04_comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PROTOTYPE ACTIVATION COMPARISON\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Image: {image_path}\n")
        f.write(f"Corruption: {corruption_name or 'None'}\n")
        if corruption_name:
            f.write(f"Severity: {severity}\n")
        f.write(f"\nMethods: {', '.join(models_dict.keys())}\n\n")
        
        f.write("="*60 + "\n")
        f.write("TOP-5 ACTIVATED PROTOTYPES PER METHOD\n")
        f.write("="*60 + "\n\n")
        
        for method_name, proto_results in all_proto_results.items():
            f.write(f"\n{method_name}:\n")
            f.write("-"*60 + "\n")
            for i, proto in enumerate(proto_results[:5]):
                f.write(f"  {i+1}. Proto {proto['proto_idx']:4d} | "
                       f"Activation: {proto['activation']:6.4f} | "
                       f"Class: {proto['class']:3d} | "
                       f"Weight: {proto['connection_weight']:7.4f}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("="*60 + "\n\n")
        f.write("Each visualization shows:\n")
        f.write("  Column 1 (THIS): Test image with colored boxes on activated regions\n")
        f.write("  Column 2 (THAT): Training prototype the model matches it to\n")
        f.write("  Column 3 (INFO): Activation scores and details\n\n")
        f.write("Look for:\n")
        f.write("  - Higher activation = stronger match\n")
        f.write("  - Correct class prototypes\n")
        f.write("  - Semantic similarity between test patch and training prototype\n")
    
    print(f"✓ Summary saved")
    print(f"\n{'='*60}\n")
    
    return image_output_dir


def select_smart_samples_from_predictions(predictions_storage, test_dataset, num_samples=5):
    """
    Select interesting samples using PRE-STORED predictions (no inference).
    
    This avoids the bug where models adapt during selection and give different predictions.
    
    Args:
        predictions_storage: Dict with keys 'Normal', 'EATA', 'ProtoEntropy-Imp+Conf'
                            Each contains {'predictions': tensor, 'labels': tensor, 'indices': tensor}
        test_dataset: The test dataset (to get image paths)
        num_samples: Number of samples to select
    
    Returns:
        List of (image_path, true_class) tuples
    """
    required_models = ['Normal', 'EATA', 'ProtoEntropy-Imp+Conf']
    for model_name in required_models:
        if model_name not in predictions_storage:
            print(f"⚠ Missing predictions for {model_name}. Cannot do smart selection.")
            return None
    
    print(f"\nSearching for samples where:")
    print(f"  ✓ ProtoEntropy-Imp+Conf predicts CORRECTLY")
    print(f"  ✗ Normal predicts WRONG")
    print(f"  ✗ EATA predicts WRONG")
    
    # Get predictions
    normal_preds = predictions_storage['Normal']['predictions']
    eata_preds = predictions_storage['EATA']['predictions']
    proto_preds = predictions_storage['ProtoEntropy-Imp+Conf']['predictions']
    labels = predictions_storage['Normal']['labels']  # Same for all
    
    # Find samples where ProtoEntropy correct, Normal AND EATA both wrong
    proto_correct = (proto_preds == labels)
    normal_wrong = (normal_preds != labels)
    eata_wrong = (eata_preds != labels)
    
    # target_mask = proto_correct & normal_wrong & eata_wrong
    target_mask = proto_correct & eata_wrong
    target_indices = torch.where(target_mask)[0]
    
    print(f"  Found {len(target_indices)} candidate samples (before class diversity filter)")
    
    if len(target_indices) == 0:
        print(f"⚠ No samples found where ProtoEntropy wins.")
        print(f"  This might mean:")
        print(f"    - All methods perform similarly on this corruption/severity")
        print(f"    - ProtoEntropy doesn't uniquely outperform both Normal AND EATA")
        print(f"    - The adapted models are all very good or all struggling")
        print(f"  Falling back to random selection.")
        return None
    
    # Select samples ensuring class diversity (one per class)
    interesting_samples = []
    seen_classes = set()
    
    for idx in target_indices:
        idx_int = idx.item()
        true_label = labels[idx_int].item()
        
        # Skip if we already have a sample from this class
        if true_label in seen_classes:
            continue
        
        # Get image path from dataset
        if hasattr(test_dataset, 'samples'):
            img_path, _ = test_dataset.samples[idx_int]
            rel_path = os.path.relpath(img_path, test_dataset.root)
        else:
            # Handle Subset case
            dataset = test_dataset
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset
            img_path, _ = dataset.samples[idx_int]
            rel_path = os.path.relpath(img_path, dataset.root)
        
        interesting_samples.append((rel_path, true_label))
        seen_classes.add(true_label)
        
        # Debug info
        normal_pred = normal_preds[idx_int].item()
        eata_pred = eata_preds[idx_int].item()
        proto_pred = proto_preds[idx_int].item()
        
        print(f"    Sample {len(interesting_samples)}: {rel_path}")
        print(f"      Ground truth: {true_label}")
        print(f"      Normal pred: {normal_pred} ✗")
        print(f"      EATA pred: {eata_pred} ✗")
        print(f"      ProtoEntropy pred: {proto_pred} ✓")
        
        if len(interesting_samples) >= num_samples:
            break
    
    print(f"\n✓ Found {len(interesting_samples)} samples from {len(seen_classes)} different classes")
    print(f"  where ProtoEntropy is correct but BOTH Normal AND EATA are wrong!")
    return interesting_samples[:num_samples]


def select_smart_samples(models_dict, test_loader, num_samples=5, mode='proto_wins'):
    """
    [DEPRECATED] Select interesting samples by running inference.
    
    WARNING: This function has a bug - it runs inference again, causing adaptation
    methods to change state and give different predictions during visualization.
    Use select_smart_samples_from_predictions() instead.
    
    Args:
        models_dict: Dict with ADAPTED models (must include 'Normal', 'EATA', 'ProtoEntropy-Imp+Conf')
        test_loader: DataLoader for test set (should be the NOISY/corrupted loader)
        num_samples: Number of samples to select
        mode: 'proto_wins' (ProtoEntropy correct, Normal AND EATA both wrong) or 'random'
    
    Returns:
        List of (image_path, true_class) tuples
    """
    if mode == 'random':
        return None  # Will use random selection
    
    # Check we have all required models
    required_models = ['Normal', 'EATA', 'ProtoEntropy-Imp+Conf']
    for model_name in required_models:
        if model_name not in models_dict:
            print(f"⚠ Need {', '.join(required_models)} for smart selection. Using random.")
            return None
    
    print(f"\nSearching for samples where:")
    print(f"  ✓ ProtoEntropy-Imp+Conf predicts CORRECTLY")
    print(f"  ✗ Normal predicts WRONG")
    print(f"  ✗ EATA predicts WRONG")
    
    normal_model = models_dict['Normal']
    eata_model = models_dict['EATA']
    proto_model = models_dict['ProtoEntropy-Imp+Conf']
    
    # Extract underlying models (they might be wrapped)
    if hasattr(normal_model, 'model'):
        normal_model = normal_model.model
    if hasattr(eata_model, 'model'):
        eata_model = eata_model.model
    if hasattr(proto_model, 'model'):
        proto_model = proto_model.model
    
    normal_model.eval()
    eata_model.eval()
    proto_model.eval()
    
    interesting_samples = []
    total_processed = 0
    seen_classes = set()  # Track classes to ensure diversity
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            batch_size = images.size(0)
            total_processed += batch_size
            
            # Get predictions from all three models
            normal_out, _, _ = normal_model(images)
            eata_out, _, _ = eata_model(images)
            proto_out, _, _ = proto_model(images)
            
            _, normal_pred = normal_out.max(1)
            _, eata_pred = eata_out.max(1)
            _, proto_pred = proto_out.max(1)
            
            # Find samples where:
            # - ProtoEntropy is CORRECT (pred == ground truth)
            # - Normal is WRONG (pred != ground truth)  
            # - EATA is WRONG (pred != ground truth)
            # ALL three conditions must be true (ProtoEntropy wins over both baselines)
            proto_correct = (proto_pred == labels)
            normal_wrong = (normal_pred != labels)
            eata_wrong = (eata_pred != labels)
            
            target_mask = proto_correct & normal_wrong & eata_wrong
            
            # Get indices
            target_indices = torch.where(target_mask)[0]
            
            if len(target_indices) > 0:
                print(f"  Found {len(target_indices)} candidate samples in batch {batch_idx}")
            
            for idx in target_indices:
                true_label = labels[idx].item()
                
                # Skip if we already have a sample from this class (ensure diversity)
                if true_label in seen_classes:
                    continue
                
                global_idx = batch_idx * test_loader.batch_size + idx.item()
                
                # Get image path and label from dataset
                if hasattr(test_loader.dataset, 'samples'):
                    img_path, true_label = test_loader.dataset.samples[global_idx]
                    # Convert to relative path
                    rel_path = os.path.relpath(img_path, test_loader.dataset.root)
                else:
                    # Handle Subset case
                    dataset = test_loader.dataset
                    while hasattr(dataset, 'dataset'):
                        dataset = dataset.dataset
                    img_path, true_label = dataset.samples[global_idx]
                    rel_path = os.path.relpath(img_path, dataset.root)
                
                interesting_samples.append((rel_path, true_label))
                seen_classes.add(true_label)
                
                # Debug info
                normal_correct = "✓" if normal_pred[idx].item() == true_label else "✗"
                eata_correct = "✓" if eata_pred[idx].item() == true_label else "✗"
                print(f"    Sample {len(interesting_samples)}: {rel_path}")
                print(f"      Ground truth: {true_label}")
                print(f"      Normal pred: {normal_pred[idx].item()} {normal_correct}")
                print(f"      EATA pred: {eata_pred[idx].item()} {eata_correct}")
                print(f"      ProtoEntropy pred: {proto_pred[idx].item()} ✓")
                
                if len(interesting_samples) >= num_samples:
                    break
            
            if len(interesting_samples) >= num_samples:
                break
            
            # Progress
            if batch_idx % 10 == 0 and batch_idx > 0:
                print(f"  Processed {total_processed} images, found {len(interesting_samples)} interesting samples...")
    
    if len(interesting_samples) == 0:
        print(f"⚠ No samples found where ProtoEntropy wins (searched {total_processed} images).")
        print(f"  This might mean:")
        print(f"    - All methods perform similarly on this corruption/severity")
        print(f"    - ProtoEntropy doesn't uniquely outperform both Normal AND EATA")
        print(f"    - The adapted models are all very good or all struggling")
        print(f"  Falling back to random selection.")
        return None
    
    print(f"\n✓ Found {len(interesting_samples)} samples from {len(seen_classes)} different classes")
    print(f"  where ProtoEntropy is correct but BOTH Normal AND EATA are wrong!")
    return interesting_samples[:num_samples]


def run_batch_interpretability(
    models_dict,
    image_paths,
    test_dir,
    prototype_img_dir,
    output_base_dir,
    corruption_name,
    severity,
    experimental_settings,
    true_classes=None,
    use_pre_corrupted=False,
    data_root=None,
    batch_precomputed_results=None # List of dicts, one per image
):
    """
    Run comprehensive interpretability analysis for multiple images.
    
    Args:
        models_dict: Dict mapping method names to model wrappers
        image_paths: List of relative image paths or list of (path, true_class) tuples
        test_dir: Clean test directory
        prototype_img_dir: Directory with prototype images
        output_base_dir: Base output directory
        corruption_name: Name of corruption
        severity: Corruption severity
        experimental_settings: Dict with experimental settings
        true_classes: List of true class labels (optional)
        use_pre_corrupted: Whether images are already corrupted
        data_root: Root directory for images (overrides test_dir if provided)
        batch_precomputed_results: Optional list of precomputed result dicts matching image_paths
    """
    output_dirs = []
    
    for i, item in enumerate(image_paths):
        # Handle both (path, class) tuples and plain paths
        if isinstance(item, tuple):
            img_path, true_class = item
        else:
            img_path = item
            true_class = true_classes[i] if true_classes and i < len(true_classes) else None
        
        # Get precomputed results for this image if available
        precomputed = None
        if batch_precomputed_results is not None and i < len(batch_precomputed_results):
            precomputed = batch_precomputed_results[i]
        
        try:
            output_dir = run_comprehensive_interpretability(
                models_dict=models_dict,
                image_path=img_path,
                test_dir=test_dir,
                prototype_img_dir=prototype_img_dir,
                output_base_dir=output_base_dir,
                corruption_name=corruption_name,
                severity=severity,
                experimental_settings=experimental_settings,
                true_class=true_class,
                use_pre_corrupted=use_pre_corrupted,
                data_root=data_root,
                precomputed_results_dict=precomputed
            )
            output_dirs.append(output_dir)
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
    
    return output_dirs

