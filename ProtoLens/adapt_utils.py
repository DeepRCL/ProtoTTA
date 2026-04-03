"""
Utilities for test-time adaptation parameter collection.
Supports different adaptation modes for ProtoLens.
"""

import torch.nn as nn


def collect_params(model, adaptation_mode='layernorm_only'):
    """Collect parameters for test-time adaptation.
    
    Args:
        model: The ProtoLens model (BERTClassifier)
        adaptation_mode: What to adapt during TTA
            'layernorm_only' - Only LayerNorm/BatchNorm (default, safest)
            'layernorm_attn_bias' - LayerNorms + Attention biases (TENT & ProtoTTA V3)
    
    Returns:
        params: List of parameters to optimize
        names: List of parameter names
    """
    params = []
    names = []
    
    # Get BERT model from SentenceTransformer
    bert_model = model.bert._first_module().auto_model
    
    # Always include LayerNorm/BatchNorm if mode includes 'layernorm'
    if 'layernorm' in adaptation_mode:
        # Collect from BERT backbone
        for nm, m in bert_model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"bert._first_module().auto_model.{nm}.{np}")
        
        # Also collect from ProtoLens components (if any)
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                # Skip if already collected from BERT
                if 'bert' not in nm:
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:
                            params.append(p)
                            names.append(f"{nm}.{np}")
    
    # Add attention biases from BERT backbone
    if 'attn_bias' in adaptation_mode:
        for nm, m in bert_model.named_modules():
            # Look for attention modules: encoder.layer.X.attention.attn.{q,k,v,o}.bias
            if 'attention' in nm and 'attn' in nm:
                for np, p in m.named_parameters():
                    if 'bias' in np and np in ['q.bias', 'k.bias', 'v.bias', 'o.bias']:
                        params.append(p)
                        names.append(f"bert._first_module().auto_model.{nm}.{np}")
    
    return params, names


def configure_model(model, adaptation_mode='layernorm_only'):
    """Configure model for test-time adaptation.
    
    Args:
        model: The ProtoLens model (BERTClassifier)
        adaptation_mode: What parameters to enable for adaptation
    
    IMPORTANT: We keep the model in eval mode to disable dropout, but set 
    requires_grad on specific parameters. Unlike BatchNorm, LayerNorm behavior
    is identical in train/eval mode, so this is safe for LayerNorm adaptation.
    """
    # CRITICAL: Start in eval mode to disable dropout in BERT backbone
    # This prevents non-deterministic outputs that can cause adaptation instability
    model.eval()
    
    # Disable all gradients first
    model.requires_grad_(False)
    
    # Get BERT model from SentenceTransformer
    bert_model = model.bert._first_module().auto_model
    
    # Configure LayerNorms for adaptation
    # NOTE: LayerNorm behavior is identical in train/eval mode (no running stats).
    # We only need to enable gradients, NOT switch to train mode.
    if 'layernorm' in adaptation_mode:
        # Configure BERT LayerNorms - enable gradients only
        for m in bert_model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True
        
        # Configure ProtoLens components (if any)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # For BatchNorm, we need train mode to use batch stats
                # But switch this specific module to train mode
                m.train()
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.LayerNorm):
                # Only enable gradients, stay in eval mode
                for p in m.parameters():
                    p.requires_grad = True
    
    # Enable attention biases
    if 'attn_bias' in adaptation_mode:
        for nm, m in bert_model.named_modules():
            # Changed from AND to OR - different transformers use different naming
            if 'attention' in nm.lower() or 'attn' in nm.lower():
                for np, p in m.named_parameters():
                    if 'bias' in np.lower():
                        p.requires_grad = True
    
    return model
