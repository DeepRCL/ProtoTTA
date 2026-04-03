"""
MEMO: Test Time Robustness via Adaptation and Augmentation

This module implements the MEMO algorithm for test-time adaptation.
MEMO adapts the model to each test sample individually by minimizing
the entropy of the marginal distribution over multiple augmented views.

Paper: https://arxiv.org/abs/2110.09506
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms

from tent import copy_model_and_optimizer, load_model_and_optimizer
from preprocess import mean as IMAGENET_MEAN, std as IMAGENET_STD


class MEMO(nn.Module):
    """MEMO adapts a model by marginal entropy minimization during testing.
    
    For each test sample, MEMO:
    1. Generates multiple augmented views
    2. Passes them through the model
    3. Computes the average (marginal) probability distribution
    4. Minimizes the entropy of this marginal distribution
    5. Resets the model after each sample (episodic)
    
    Args:
        model: The model to adapt
        optimizer: Optimizer for adaptation
        steps: Number of adaptation steps per sample (default: 1)
        batch_size: Number of augmented views per adaptation step (default: 32)
        augmentation_type: Type of augmentation to use ('augmix' or 'standard')
    """
    
    def __init__(self, model, optimizer, steps=1, batch_size=32, 
                 augmentation_type='augmix', episodic=True):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.batch_size = batch_size
        self.augmentation_type = augmentation_type
        self.episodic = episodic
        
        assert steps > 0, "MEMO requires >= 1 step(s) to adapt"
        assert episodic, "MEMO is designed for episodic adaptation"
        
        # Store initial model and optimizer state for reset
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        # Setup augmentation transforms
        self._setup_augmentations()
    
    def _setup_augmentations(self):
        """Setup augmentation functions based on augmentation type."""
        if self.augmentation_type == 'augmix':
            self.aug_fn = create_augmix_augmentation()
        else:
            self.aug_fn = create_standard_augmentation()
    
    def forward(self, x):
        """Forward pass with test-time adaptation.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            outputs: Model predictions
            min_distances: Minimum distances to prototypes
            values: Prototype values
        """
        if self.episodic:
            self.reset()
        
        # Adaptation loop
        with torch.enable_grad():
            self.model.train()
            for _ in range(self.steps):
                # Generate augmented views
                aug_images = self._generate_augmented_views(x)
                
                # Forward pass on augmented views
                self.optimizer.zero_grad()
                outputs_aug, _, _ = self.model(aug_images)
                
                # Compute marginal entropy loss
                loss = marginal_entropy(outputs_aug)
                
                # Backprop and update
                loss.backward()
                self.optimizer.step()
        
        # Final prediction on original image (no grad)
        self.model.eval()
        with torch.no_grad():
            outputs, min_distances, values = self.model(x)
        
        return outputs, min_distances, values
    
    def _generate_augmented_views(self, x):
        """Generate batch_size augmented views of the input.
        
        Args:
            x: Input tensor [B, C, H, W] (already normalized)
        
        Returns:
            Augmented images tensor [batch_size, C, H, W]
        """
        B, C, H, W = x.shape
        
        # For now, we assume batch size is 1 for test-time adaptation
        assert B == 1, "MEMO processes one image at a time"
        
        # Get the single image
        img_tensor = x[0]  # [C, H, W]
        
        # Denormalize to [0, 1] range for augmentation
        img_denorm = self._denormalize(img_tensor)
        
        # Convert to PIL Image
        img_pil = transforms.ToPILImage()(img_denorm.cpu())
        
        # Create augmented views
        augmented_views = []
        for _ in range(self.batch_size):
            # Apply augmentation (returns PIL Image)
            aug_pil = self.aug_fn(img_pil)
            
            # Convert back to tensor and normalize
            aug_tensor = transforms.ToTensor()(aug_pil)
            aug_normalized = self._normalize(aug_tensor)
            
            augmented_views.append(aug_normalized)
        
        # Stack augmented views
        aug_batch = torch.stack(augmented_views, dim=0)  # [batch_size, C, H, W]
        
        return aug_batch.to(x.device)
    
    def _denormalize(self, tensor):
        """Denormalize a tensor using ImageNet statistics.
        
        Args:
            tensor: [C, H, W] normalized tensor
        
        Returns:
            Denormalized tensor in [0, 1] range
        """
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
        return tensor * std + mean
    
    def _normalize(self, tensor):
        """Normalize a tensor using ImageNet statistics.
        
        Args:
            tensor: [C, H, W] tensor in [0, 1] range
        
        Returns:
            Normalized tensor
        """
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
        return (tensor - mean) / std
    
    def reset(self):
        """Reset model and optimizer to initial state."""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    def __getattr__(self, name):
        """Forward attribute access to the underlying model if not found in MEMO."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def marginal_entropy(outputs):
    """Compute marginal entropy over a batch of predictions.
    
    The marginal distribution is the average probability distribution
    across all augmented views. Minimizing its entropy encourages
    both confidence and consistency.
    
    Args:
        outputs: Model logits [B, num_classes]
    
    Returns:
        Scalar entropy loss
    """
    # Convert logits to log probabilities
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    
    # Average log probabilities across batch (marginal distribution)
    # log(mean(p)) = logsumexp(log(p)) - log(B)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    
    # Clamp to avoid numerical issues
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    
    # Entropy: -sum(p * log(p))
    entropy = -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
    
    return entropy


def create_augmix_augmentation():
    """Create AugMix-style augmentation function.
    
    AugMix applies multiple augmentation chains and mixes them together.
    Based on: https://github.com/google-research/augmix
    
    Returns:
        Function that takes a PIL Image and returns an augmented PIL Image
    """
    
    # Define augmentation operations
    augmentations = [
        autocontrast,
        equalize,
        lambda x: rotate(x, 1),
        lambda x: solarize(x, 1),
        lambda x: posterize(x, 1),
    ]
    
    def augmix_fn(pil_img):
        """Apply AugMix augmentation to a PIL Image.
        
        Args:
            pil_img: PIL Image
        
        Returns:
            Augmented PIL Image
        """
        # Apply preaugmentation
        if np.random.rand() > 0.5:
            pil_img = transforms.RandomResizedCrop(224, scale=(0.8, 1.0))(pil_img)
        else:
            pil_img = transforms.Resize(256)(pil_img)
            pil_img = transforms.RandomCrop(224)(pil_img)
        
        if np.random.rand() > 0.5:
            pil_img = transforms.RandomHorizontalFlip(p=1.0)(pil_img)
        
        # Apply random augmentation chain
        depth = np.random.randint(1, 4)
        for _ in range(depth):
            aug_fn = np.random.choice(augmentations)
            pil_img = aug_fn(pil_img)
        
        return pil_img
    
    return augmix_fn


def create_standard_augmentation():
    """Create standard data augmentation function.
    
    Returns:
        Function that takes a PIL Image and returns an augmented PIL Image
    """
    
    def standard_aug_fn(pil_img):
        """Apply standard augmentation to a PIL Image."""
        aug_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
        ])
        return aug_transform(pil_img)
    
    return standard_aug_fn


# Augmentation helper functions (PIL-based)
def autocontrast(pil_img, level=None):
    """Apply autocontrast to PIL image."""
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, level=None):
    """Apply histogram equalization to PIL image."""
    return ImageOps.equalize(pil_img)


def rotate(pil_img, level):
    """Rotate PIL image by random angle."""
    degrees = int_parameter(rand_lvl(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR, fillcolor=128)


def solarize(pil_img, level):
    """Solarize PIL image."""
    level = int_parameter(rand_lvl(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def posterize(pil_img, level):
    """Posterize PIL image."""
    level = int_parameter(rand_lvl(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def int_parameter(level, maxval):
    """Helper function to scale level to integer parameter."""
    return int(level * maxval / 10)


def rand_lvl(n):
    """Random level for augmentation strength."""
    return np.random.uniform(low=0.1, high=n)


def configure_model(model):
    """Configure model for MEMO adaptation.
    
    MEMO can adapt all model parameters or just normalization layers.
    Here we follow the common practice of adapting all parameters.
    """
    model.train()
    model.requires_grad_(True)
    return model


def collect_params(model):
    """Collect all trainable parameters from model.
    
    For MEMO, we typically adapt all parameters, but can be restricted
    to specific layers if needed.
    """
    params = []
    names = []
    for nm, p in model.named_parameters():
        if p.requires_grad:
            params.append(p)
            names.append(nm)
    return params, names


def setup_memo(model, lr=0.00025, batch_size=64, steps=1):
    """Setup MEMO adaptation wrapper.
    
    Args:
        model: The model to adapt
        lr: Learning rate for adaptation
        batch_size: Number of augmented views per step
        steps: Number of adaptation steps per sample
    
    Returns:
        MEMO-wrapped model
    """
    # Configure model
    model = configure_model(model)
    
    # Collect parameters to adapt
    params, param_names = collect_params(model)
    
    # Setup optimizer
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    
    # Wrap with MEMO
    memo_model = MEMO(model, optimizer, steps=steps, 
                      batch_size=batch_size, episodic=True)
    
    return memo_model

