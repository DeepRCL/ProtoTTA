"""
Test utilities for ProtoPNet TTA evaluation.

This module provides functions for evaluating ProtoPNet models,
computing accuracy and loss components.
"""

import torch
import torch.nn.functional as F


def test(model, dataloader, class_specific=True, log=print, clst_k=1, sum_cls=False):
    """
    Evaluate model on a dataloader.

    Args:
        model: ProtoPNet model (possibly wrapped in TTA adapter)
        dataloader: Test data loader
        class_specific: Whether to use class-specific prototype evaluation
        log: Logging function
        clst_k: Top-k for cluster loss computation
        sum_cls: Whether to sum class contributions

    Returns:
        accuracy: Test accuracy
        loss_dict: Dictionary of loss components
    """
    model.eval()

    n_examples = 0
    n_correct = 0
    total_cross_entropy = 0.0
    total_cluster_cost = 0.0
    total_separation_cost = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()

            # Forward pass
            outputs = model(images)

            # Handle different output formats
            if isinstance(outputs, tuple):
                if len(outputs) >= 2:
                    logits, min_distances = outputs[0], outputs[1]
                else:
                    logits = outputs[0]
                    min_distances = None
            else:
                logits = outputs
                min_distances = None

            # Compute cross-entropy
            cross_entropy = F.cross_entropy(logits, labels)

            # Compute cluster and separation costs if we have distances
            if min_distances is not None:
                # Get model for prototype info
                base_model = get_base_model(model)

                if hasattr(base_model, 'core'):
                    prototype_class_identity = base_model.core.prototype_class_identity
                elif hasattr(base_model, 'prototype_class_identity'):
                    prototype_class_identity = base_model.prototype_class_identity
                else:
                    prototype_class_identity = None

                if prototype_class_identity is not None:
                    cluster_cost, separation_cost = compute_cluster_separation_costs(
                        min_distances, labels, prototype_class_identity
                    )
                else:
                    cluster_cost = torch.tensor(0.0)
                    separation_cost = torch.tensor(0.0)
            else:
                cluster_cost = torch.tensor(0.0)
                separation_cost = torch.tensor(0.0)

            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            correct = (predicted == labels).sum().item()

            n_correct += correct
            n_examples += labels.size(0)
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            n_batches += 1

    accuracy = n_correct / n_examples

    loss_dict = {
        'cross_entropy': total_cross_entropy / n_batches,
        'cluster_cost': total_cluster_cost / n_batches,
        'separation_cost': total_separation_cost / n_batches,
    }

    log(f"Test accuracy: {accuracy * 100:.2f}%")
    log(f"  Cross-entropy: {loss_dict['cross_entropy']:.4f}")
    log(f"  Cluster cost: {loss_dict['cluster_cost']:.4f}")
    log(f"  Separation cost: {loss_dict['separation_cost']:.4f}")

    return accuracy, loss_dict


def compute_cluster_separation_costs(min_distances, labels, prototype_class_identity):
    """
    Compute cluster and separation cost components.

    Args:
        min_distances: Tensor of shape (batch_size, num_prototypes)
        labels: Tensor of shape (batch_size,)
        prototype_class_identity: Tensor of shape (num_prototypes, num_classes)

    Returns:
        cluster_cost: Mean distance to closest same-class prototype
        separation_cost: Mean distance to closest different-class prototype
    """
    batch_size = min_distances.size(0)
    num_prototypes = min_distances.size(1)

    # Get prototype class assignments
    # prototype_class_identity[p, c] = 1 if prototype p belongs to class c

    cluster_costs = []
    separation_costs = []

    for i in range(batch_size):
        label = labels[i]
        distances = min_distances[i]

        # Get prototypes for the same class
        same_class_mask = prototype_class_identity[:, label].bool()

        if same_class_mask.sum() > 0:
            same_class_distances = distances[same_class_mask]
            cluster_cost = same_class_distances.min()
            cluster_costs.append(cluster_cost)

        # Get prototypes for different classes
        diff_class_mask = ~same_class_mask

        if diff_class_mask.sum() > 0:
            diff_class_distances = distances[diff_class_mask]
            separation_cost = diff_class_distances.min()
            separation_costs.append(separation_cost)

    if cluster_costs:
        mean_cluster = torch.stack(cluster_costs).mean()
    else:
        mean_cluster = torch.tensor(0.0, device=min_distances.device)

    if separation_costs:
        mean_separation = torch.stack(separation_costs).mean()
    else:
        mean_separation = torch.tensor(0.0, device=min_distances.device)

    return mean_cluster, mean_separation


def get_base_model(model):
    """
    Extract base ProtoPNet model from TTA wrapper.

    Handles various wrapper types: Tent, ProtoEntropy, EATA, etc.
    """
    # Check if it's a wrapper with .model attribute
    if hasattr(model, 'model'):
        return get_base_model(model.model)

    # Check for DataParallel
    if hasattr(model, 'module'):
        return get_base_model(model.module)

    return model


def evaluate_with_details(model, dataloader, log=print):
    """
    Detailed evaluation with per-class accuracy breakdown.

    Args:
        model: Model to evaluate
        dataloader: Test data loader
        log: Logging function

    Returns:
        accuracy: Overall accuracy
        class_accuracies: Dict mapping class_idx -> accuracy
        confusion_matrix: Confusion matrix tensor
    """
    model.eval()

    # Get number of classes
    base_model = get_base_model(model)
    if hasattr(base_model, 'core'):
        num_classes = base_model.core.num_classes
    elif hasattr(base_model, 'num_classes'):
        num_classes = base_model.num_classes
    else:
        # Infer from dataloader
        num_classes = len(dataloader.dataset.classes) if hasattr(dataloader.dataset, 'classes') else 5

    # Initialize counters
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    confusion = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            _, predicted = torch.max(logits, 1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()

                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

                confusion[label, pred] += 1

    # Compute accuracies
    overall_accuracy = class_correct.sum().item() / class_total.sum().item()
    class_accuracies = {}

    for c in range(num_classes):
        if class_total[c] > 0:
            class_accuracies[c] = class_correct[c].item() / class_total[c].item()
        else:
            class_accuracies[c] = 0.0

    # Log results
    log(f"\nOverall accuracy: {overall_accuracy * 100:.2f}%")
    log("\nPer-class accuracy:")
    class_names = dataloader.dataset.classes if hasattr(dataloader.dataset, 'classes') else [str(i) for i in range(num_classes)]
    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else str(c)
        log(f"  {name}: {class_accuracies[c] * 100:.2f}% ({int(class_correct[c])}/{int(class_total[c])})")

    return overall_accuracy, class_accuracies, confusion
