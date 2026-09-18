"""
Preprocessing utilities for ProtoPNet TTA.

Contains normalization statistics and preprocessing functions
for the SICAPv2 histopathology dataset.
"""

import torchvision.transforms as transforms

# ImageNet normalization statistics (used for pretrained VGG backbone)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Create normalization transform
normalize = transforms.Normalize(mean=mean, std=std)


def preprocess_input_function(x):
    """
    Normalize tensor for pretrained model.

    Args:
        x: Tensor of shape (N, C, H, W) with values in [0, 1]

    Returns:
        Normalized tensor
    """
    return normalize(x)


def get_preprocess_transform(img_size=224, include_normalize=True):
    """
    Get preprocessing transform pipeline.

    Args:
        img_size: Target image size
        include_normalize: Whether to include normalization

    Returns:
        Composed transform
    """
    transform_list = [
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]

    if include_normalize:
        transform_list.append(normalize)

    return transforms.Compose(transform_list)
