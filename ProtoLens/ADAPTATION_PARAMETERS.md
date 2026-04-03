# ProtoLens Adaptation Parameters

## Model Architecture

ProtoLens uses **SentenceTransformer (all-mpnet-base-v2)** which is based on **MPNet** (BERT-like architecture).

### BERT/MPNet Structure:
- **25 LayerNorm modules**: 
  - 1 in embeddings
  - 12 layers × 2 LayerNorms per layer (attention + output) = 24
- **48 Attention biases**:
  - 12 layers × 4 biases per layer (q, k, v, o) = 48

## Parameters Adapted

### All Methods (TENT, EATA, ProtoTTA):
**Adaptation Mode**: `layernorm_attn_bias`

1. **LayerNorm parameters** (25 modules × 2 params = 50 params):
   - `weight` (scale)
   - `bias` (shift)
   - From BERT backbone: `encoder.layer.X.attention.LayerNorm` and `encoder.layer.X.output.LayerNorm`

2. **Attention biases** (48 params):
   - `encoder.layer.X.attention.attn.q.bias` (12 params)
   - `encoder.layer.X.attention.attn.k.bias` (12 params)
   - `encoder.layer.X.attention.attn.v.bias` (12 params)
   - `encoder.layer.X.attention.attn.o.bias` (12 params)

**Total**: ~98 parameters adapted (~0.0003% of model)

## Comparison with ProtoViT

| Aspect | ProtoViT | ProtoLens |
|--------|----------|-----------|
| **Backbone** | Vision Transformer (ViT) | MPNet (BERT-like) |
| **Normalization** | LayerNorm only | LayerNorm only |
| **Attention** | Self-attention (no biases) | Self-attention with biases |
| **TENT adapts** | LayerNorm only | LayerNorm + Attention biases ✅ |
| **ProtoTTA adapts** | LayerNorm + Attention biases | LayerNorm + Attention biases ✅ |

## Why Attention Biases?

**Attention biases** control where the model focuses in the input sequence:
- **Q/K/V biases**: Affect query/key/value computation
- **O bias**: Affects output projection

Under distribution shift (Yelp → Hotel), attention patterns need adjustment to:
- Focus on relevant tokens
- Ignore domain-specific noise
- Restore semantic focus

This is especially important for prototype-based models that rely on meaningful attention patterns.

## Implementation Details

### Parameter Collection (`adapt_utils.py`)

```python
# Collects from BERT backbone:
bert_model = model.bert._first_module().auto_model

# LayerNorms
for nm, m in bert_model.named_modules():
    if isinstance(m, nn.LayerNorm):
        # Collect weight and bias

# Attention biases  
for nm, m in bert_model.named_modules():
    if 'attention' in nm and 'attn' in nm:
        if 'bias' in np and np in ['q.bias', 'k.bias', 'v.bias', 'o.bias']:
            # Collect bias
```

### Configuration (`configure_model`)

1. Set model to `train()` mode
2. Disable all gradients: `model.requires_grad_(False)`
3. Enable LayerNorm gradients
4. Enable attention bias gradients

## Verification

To verify parameters are collected correctly:

```python
from adapt_utils import collect_params, configure_model

model = BERTClassifier(...)
model = configure_model(model, adaptation_mode='layernorm_attn_bias')
params, names = collect_params(model, adaptation_mode='layernorm_attn_bias')

print(f"Total parameters: {len(params)}")
print(f"LayerNorms: {sum('LayerNorm' in n for n in names)}")
print(f"Attention biases: {sum('bias' in n and 'attn' in n for n in names)}")
```

Expected output:
```
Total parameters: 98
LayerNorms: 50 (25 modules × 2 params)
Attention biases: 48 (12 layers × 4 biases)
```
