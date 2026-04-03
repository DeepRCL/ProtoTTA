# ProtoTTA: Prototype-Aware Test-Time Adaptation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ProtoTTA** (Prototype-Aware Test-Time Adaptation) is a test-time adaptation framework specifically designed for prototype-based neural networks. Unlike standard TTA methods that rely solely on output logits, ProtoTTA leverages intermediate prototype signals to achieve more effective adaptation under distribution shifts.

<img width="2809" height="1453" alt="Picture2 (1)" src="https://github.com/user-attachments/assets/381a6f08-3197-4207-9a0a-0f0092eea5a7" />


## Overview

ProtoTTA minimizes the binary entropy of prototype-similarity distributions, encouraging decisive and semantically meaningful activations. Key features include:

- **Robust consensus aggregation**: Uses top-k mean of sub-prototypes instead of maximum to reduce sensitivity to outliers
- **Geometric filtering**: Restricts updates to samples with reliable prototype activations
- **Prototype-aware adaptation**: Updates attention biases and LayerNorm parameters to restore semantic focus
- **Stability mechanisms**: Prototype-importance weighting and confidence-based regularization

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch 2.x with CUDA support

### Setup

```bash
# Clone the repository
git clone https://github.com/aboots/prototype-TTA.git
cd prototype-TTA/ProtoViT

# Create conda environment
conda create -n prototta python=3.10 -y
conda activate prototta

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install robustness evaluation dependencies (for corruption generation)
pip install -r robustness_requirements.txt

# Note: For full corruption support, install ImageMagick system package:
# Ubuntu/Debian: sudo apt-get install libmagickwand-dev imagemagick
# macOS: brew install imagemagick
# (Basic corruptions work without ImageMagick, but motion_blur and snow require it)
```

## Using ProtoTTA

ProtoTTA is implemented as a standalone module (`proto_entropy.py`) that can be integrated into any prototype-based model. The core class is `ProtoEntropy`, which wraps your model and performs test-time adaptation.

### Model Requirements

Your prototype-based model must satisfy the following interface:

1. **Forward method**: `model(x)` must return a tuple `(logits, min_distances, similarities)`
   - `logits`: [batch_size, num_classes] - classification logits
   - `min_distances`: [batch_size, num_prototypes] - distance to each prototype
   - `similarities`: [batch_size, num_prototypes] or [batch_size, num_prototypes, num_sub_prototypes] - similarity scores

2. **Required attributes**:
   - `model.prototype_class_identity`: [num_prototypes, num_classes] tensor mapping prototypes to classes
   - `model.last_layer`: Classification head (required for prototype importance weighting)

3. **Optional attributes** (depending on `adaptation_mode`):
   - `model.prototype_vectors`: Prototype vectors (if adapting prototypes)
   - `model.patch_select`: Patch selection parameters (if adapting patch selection)

### Basic Integration

```python
from proto_entropy import ProtoEntropy, configure_model, collect_params
import torch.optim as optim

# 1. Configure model for adaptation
# This enables gradients for the specified components
model = configure_model(model, adaptation_mode='layernorm_attn_bias')

# 2. Collect parameters to adapt
params, param_names = collect_params(model, adaptation_mode='layernorm_attn_bias')
print(f"Adapting parameters: {param_names}")

# 3. Setup optimizer
optimizer = optim.Adam(params, lr=0.001)

# 4. Create ProtoTTA wrapper (using best configuration from experiments)
proto_tta = ProtoEntropy(
    model=model,
    optimizer=optimizer,
    steps=1,  # Number of adaptation steps per batch
    use_geometric_filter=True,  # Filter unreliable samples
    geo_filter_threshold=0.92,  # Similarity threshold for filtering
    consensus_strategy='top_k_mean',  # How to aggregate sub-prototypes
    use_prototype_importance=True,  # Weight by prototype importance
    use_confidence_weighting=True,  # Weight by prediction confidence
    use_ensemble_entropy=False,  # Best: aggregate first, then compute entropy
    adaptation_mode='layernorm_attn_bias'  # What to adapt (best for ProtoViT)
)

# 5. During inference (ProtoTTA adapts automatically)
logits, min_distances, similarities = proto_tta(x)
```

### Adaptation Modes

The `adaptation_mode` parameter controls which model components are adapted:

- `layernorm_only`: Only LayerNorm/BatchNorm parameters (safest, default)
- `layernorm_attn_bias`: LayerNorms + Attention biases (recommended for ProtoViT)
- `layernorm_proto`: LayerNorms + Prototype vectors
- `layernorm_proto_patch`: LayerNorms + Prototypes + Patch selection
- `layernorm_proto_last`: LayerNorms + Prototypes + Last layer
- `full_proto`: Prototypes + Patch selection + Last layer (no backbone)
- `all_adaptive`: Everything except frozen backbone features

## Dataset Preparation

### CUB-200-2011 Dataset

1. Download [CUB_200_2011.tgz](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
2. Extract and preprocess:
   ```bash
   tar -xzf CUB_200_2011.tgz
   # Follow dataset instructions for cropping and train/test split
   # Place cropped images in ./datasets/cub200_cropped/
   ```
3. Augment training data:
   ```bash
   python img_aug.py
   ```

### Creating CUB-200-C (Corrupted Dataset)

To evaluate robustness, create corrupted versions of the test set:

```bash
python create_cub_c.py \
    --input_dir ./datasets/cub200_cropped/test_cropped/ \
    --output_dir ./datasets/cub200_c/ \
    --corruption all \
    --severity 1,2,3,4,5
```

This creates corrupted datasets for all corruption types at specified severity levels.

## Training

1. Configure dataset paths in `settings.py`:
   ```python
   data_path = "./datasets/cub200_cropped/"
   train_dir = data_path + "train_cropped_augmented/"
   test_dir = data_path + "test_cropped/"
   ```

2. Train the model:
   ```bash
   python main.py
   ```

## Evaluation

### Test-Time Adaptation Inference

Run ProtoTTA on corrupted test data:

```bash
python run_inference.py \
    -corruption gaussian_noise \
    -severity 5 \
    -mode proto_importance_confidence \
    --use-geometric-filter \
    --geo-filter-threshold 0.92 \
    --consensus-strategy top_k_mean \
    --adaptation-mode layernorm_attn_bias
```

**Note:** The best configuration (achieving 60.09% mean accuracy on CUB-200-C) uses:
- `use_ensemble_entropy=False` (aggregate sub-prototypes first, then compute entropy)
- `use_geometric_filter=True` with `geo_filter_threshold=0.92`
- `consensus_strategy='top_k_mean'`
- `adaptation_mode='layernorm_attn_bias'`

**Key Arguments:**
- `-mode`: Adaptation method (`normal`, `eata`, `proto_importance_confidence`, or comma-separated list)
- `--use-geometric-filter`: Enable geometric filtering for reliable samples (recommended: True)
- `--geo-filter-threshold`: Similarity threshold for filtering (best: 0.92)
- `--consensus-strategy`: How to aggregate sub-prototypes (`max`, `mean`, `median`, `top_k_mean`, `weighted_mean`). Best: `top_k_mean`
- `--adaptation-mode`: What to adapt (`layernorm_only`, `layernorm_attn_bias`, `layernorm_proto`, etc.). Best: `layernorm_attn_bias`
- `--use-ensemble-entropy`: Use ensemble entropy across sub-prototypes (best: False - aggregate first, then compute entropy)
- `--use-source-stats`: Use source distribution regularization (optional)
- `--alpha-source-kl`: Weight for source KL regularization (default: 0.0, optional)

**Example with source statistics (optional regularization):**
```bash
python run_inference.py \
    -mode proto_importance_confidence \
    -corruption gaussian_noise \
    -severity 5 \
    --use-geometric-filter \
    --geo-filter-threshold 0.92 \
    --consensus-strategy top_k_mean \
    --adaptation-mode layernorm_attn_bias \
    --use-source-stats \
    --alpha-source-kl 0.005 \
    --num-source-samples 2000
```

**Note:** Source statistics regularization is optional and can help prevent drift, but the best results (60.09% mean accuracy) were achieved without it.

**Multiple methods comparison:**
```bash
python run_inference.py \
    -corruption gaussian_noise \
    -severity 5 \
    -mode normal,eata,proto_importance_confidence \
    --use-geometric-filter \
    --geo-filter-threshold 0.92 \
    --consensus-strategy top_k_mean \
    --adaptation-mode layernorm_attn_bias
```

### Robustness Evaluation

Evaluate model performance across multiple corruptions:

```bash
python evaluate_robustness.py \
    --model ./saved_models/best_model.pth \
    --data_dir ./datasets/cub200_c/ \
    --output ./robustness_results_sev5.json
```

## Visualization and Analysis

### Robustness Results Visualization

Visualize and analyze robustness evaluation results:

```bash
python visualize_robustness_results.py \
    --input robustness_results_sev5.json \
    --output_dir ./plots/robustness_analysis \
    --save_summary summary_stats.json
```

### Method Comparison

Compare two adaptation methods:

```bash
python compare_methods.py \
    --input robustness_results_sev5.json \
    --method1 eata \
    --method2 proto_imp_conf_v3
```

### Ablation Studies

Run ablation studies to analyze component contributions:

```bash
python run_ablation_studies.py --use_all_samples
```

Visualize ablation results:

```bash
python visualize_robustness_results.py \
    --input robustness_results_sev5.json \
    --csv_input ./ablation_studies/visualizations/detailed_results.csv
```

### Interpretability Visualization

Generate interpretability visualizations:

```bash
python run_inference.py \
    -corruption gaussian_noise \
    -severity 5 \
    -mode proto_importance_confidence \
    --use-geometric-filter \
    --geo-filter-threshold 0.92 \
    --consensus-strategy top_k_mean \
    --adaptation-mode layernorm_attn_bias \
    --interpretability-num-images 5 \
    --interpretability-mode proto_wins
```

**Best Configuration Summary:**
Based on experimental results on CUB-200-C (severity 5), the best configuration achieving **60.09% mean accuracy** (vs 51.89% baseline) is:
- `use_geometric_filter=True`, `geo_filter_threshold=0.92`
- `consensus_strategy='top_k_mean'`, `consensus_ratio=0.5`
- `adaptation_mode='layernorm_attn_bias'`
- `use_prototype_importance=True`, `use_confidence_weighting=True`
- `use_ensemble_entropy=False` (aggregate first, then compute entropy)

### Class Prototype Visualization

Visualize learned class prototypes:

```bash
python visualize_class_prototypes.py
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `run_inference.py` | Main inference script with TTA support |
| `evaluate_robustness.py` | Comprehensive robustness evaluation across corruptions |
| `create_cub_c.py` | Generate corrupted CUB-200-C dataset |
| `visualize_robustness_results.py` | Visualize and analyze robustness results |
| `compare_methods.py` | Compare different adaptation methods |
| `run_ablation_studies.py` | Run ablation studies |
| `visualize_class_prototypes.py` | Visualize learned prototypes |
| `interpretability_viz.py` | Generate interpretability visualizations |

## Citation

If you use ProtoTTA in your research, please cite:

```bibtex
coming soon
```

## Contact

For questions or issues, please contact: mahdi.abootorabi@ece.ubc.ca
