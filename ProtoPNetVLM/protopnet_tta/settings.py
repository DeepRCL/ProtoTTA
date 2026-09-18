"""
Settings for ProtoPNet TTA on SICAPv2 dataset.

This file contains configuration parameters for test-time adaptation
experiments on the SICAPv2 histopathology dataset.
"""

# Model architecture (matching training config)
base_architecture = 'vgg19'
img_size = 224

# Prototype configuration (will be loaded from model checkpoint)
# Default VGG19 produces 128-channel features
prototype_depth = 128

# SICAPv2 dataset has 5 classes: NC, G3, G4, G4C, G5
num_classes = 5

# Prototype activation function
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

# Data paths - SICAPv2 dataset
data_path = "./datasets/SICAPv2_cropped/"
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'

# Batch sizes
train_batch_size = 80
test_batch_size = 80
train_push_batch_size = 75

# TTA-specific settings
# For prototype-entropy computations
k = 1  # Top-k prototypes to consider
sum_cls = False  # Whether to sum class contributions

# Evaluation settings
experiment_run = 'sicapv2_001'
