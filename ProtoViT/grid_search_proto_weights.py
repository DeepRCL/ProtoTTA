import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import train_and_test as tnt
import model
import proto_entropy
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform
from pathlib import Path
import torch.optim as optim
import numpy as np
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Cfg:
    def __init__(self):
        self.OPTIM = self.Optim()
        self.MODEL = self.Model()

    class Optim:
        def __init__(self):
            self.METHOD = 'Adam'
            self.LR = 0.001
            self.BETA = 0.9
            self.WD = 0.000
            self.STEPS = 1

    class Model:
        def __init__(self):
            self.EPISODIC = False

cfg = Cfg()

def setup_optimizer(params):
    """Set up optimizer for adaptation."""
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError

def setup_proto_entropy_custom(model, alpha_target, alpha_separation, alpha_coherence, 
                              use_prototype_importance=False):
    """Set up ProtoEntropy adaptation with custom weights."""
    model = proto_entropy.configure_model(model)
    params, param_names = proto_entropy.collect_params(model)
    optimizer = setup_optimizer(params)
    
    # Create ProtoEntropy instance with custom weights
    proto_model = proto_entropy.ProtoEntropy(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        alpha_target=alpha_target,
        alpha_separation=alpha_separation,
        alpha_coherence=alpha_coherence,
        use_prototype_importance=use_prototype_importance
    )
    
    return proto_model

def evaluate_model(model, loader, description="Inference"):
    """Helper to run evaluation loop."""
    print(f'\n{description}...')
    class_specific = True 
    accu, test_loss_dict = tnt.test(model=model, dataloader=loader,
                                    class_specific=class_specific, log=lambda x: None, 
                                    clst_k=k, sum_cls=sum_cls)
    return accu

def run_grid_search(model_path, gpu_id='0', corruption='gaussian_noise', severity=3, 
                    use_pre_generated=True, include_baseline=True):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f'Using GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data loading
    if corruption:
        # Check if pre-generated corrupted dataset exists
        corrupted_data_dir = Path('./datasets/cub200_c')
        corruption_path = corrupted_data_dir / corruption / str(severity)
        
        if use_pre_generated and corruption_path.exists():
            print(f'Using PRE-GENERATED corrupted images from: {corruption_path}')
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            test_dataset = datasets.ImageFolder(str(corruption_path), transform)
        else:
            if use_pre_generated:
                print(f'Pre-generated corrupted images not found at {corruption_path}')
            print(f'Generating corruption ON-THE-FLY: {corruption} (Severity: {severity})')
            print(f'Loading clean images from: {test_dir}')
            transform = get_corrupted_transform(img_size, mean, std, corruption, severity)
            test_dataset = datasets.ImageFolder(test_dir, transform)
    else:
        print('Applying NO corruption (Clean Data)')
        print(f'Loading test data from: {test_dir}')
        transform = get_corrupted_transform(img_size, mean, std, None, severity)
        test_dataset = datasets.ImageFolder(test_dir, transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    print(f'Test set size: {len(test_loader.dataset)}')
    
    # Define grid - starting with a coarse grid
    # Test different combinations systematically
    grid_configs = [
        # Original (baseline with new losses)
        {'target': 1.0, 'separation': 0.5, 'coherence': 0.3, 'name': 'Default'},
        
        # No separation or coherence (original ProtoEntropy)
        {'target': 1.0, 'separation': 0.0, 'coherence': 0.0, 'name': 'Original (No Sep/Coh)'},
        
        # Only separation, no coherence
        {'target': 1.0, 'separation': 0.3, 'coherence': 0.0, 'name': 'Sep=0.3'},
        {'target': 1.0, 'separation': 0.5, 'coherence': 0.0, 'name': 'Sep=0.5'},
        {'target': 1.0, 'separation': 0.7, 'coherence': 0.0, 'name': 'Sep=0.7'},
        {'target': 1.0, 'separation': 1.0, 'coherence': 0.0, 'name': 'Sep=1.0'},
        
        # Only coherence, no separation
        {'target': 1.0, 'separation': 0.0, 'coherence': 0.3, 'name': 'Coh=0.3'},
        {'target': 1.0, 'separation': 0.0, 'coherence': 0.5, 'name': 'Coh=0.5'},
        {'target': 1.0, 'separation': 0.0, 'coherence': 0.7, 'name': 'Coh=0.7'},
        
        # Both separation and coherence - various combinations
        {'target': 1.0, 'separation': 0.3, 'coherence': 0.3, 'name': 'Sep=0.3, Coh=0.3'},
        {'target': 1.0, 'separation': 0.5, 'coherence': 0.5, 'name': 'Sep=0.5, Coh=0.5'},
        {'target': 1.0, 'separation': 0.7, 'coherence': 0.3, 'name': 'Sep=0.7, Coh=0.3'},
        {'target': 1.0, 'separation': 0.3, 'coherence': 0.7, 'name': 'Sep=0.3, Coh=0.7'},
        
        # Higher separation weight
        {'target': 1.0, 'separation': 1.0, 'coherence': 0.3, 'name': 'Sep=1.0, Coh=0.3'},
        {'target': 1.0, 'separation': 1.5, 'coherence': 0.3, 'name': 'Sep=1.5, Coh=0.3'},
        
        # Different target weights
        {'target': 0.5, 'separation': 0.5, 'coherence': 0.3, 'name': 'Tgt=0.5, Sep=0.5, Coh=0.3'},
        {'target': 1.5, 'separation': 0.5, 'coherence': 0.3, 'name': 'Tgt=1.5, Sep=0.5, Coh=0.3'},
    ]

    results = []
    best_acc = -1.0
    best_config = None

    print(f"\nRunning Grid Search for ProtoEntropy Weights")
    print(f"Corruption: {corruption} (Severity: {severity})")
    print(f"Total Configurations: {len(grid_configs)}")
    print("="*80)

    # First, evaluate baseline (no adaptation) if requested
    baseline_acc = None
    if include_baseline:
        print("\n" + "="*80)
        print("Evaluating Baseline (No Adaptation)")
        print("="*80)
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        base_model.eval()
        baseline_acc = evaluate_model(base_model, test_loader, description="Baseline")
        print(f"Baseline Accuracy: {baseline_acc*100:.2f}%")
        del base_model
        torch.cuda.empty_cache()

    for idx, config in enumerate(grid_configs):
        alpha_t = config['target']
        alpha_s = config['separation']
        alpha_c = config['coherence']
        name = config['name']
        
        print(f"\n[{idx+1}/{len(grid_configs)}] Testing: {name}")
        print(f"    Target={alpha_t}, Separation={alpha_s}, Coherence={alpha_c}")
        
        # Load fresh model for each iteration
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        
        # Setup ProtoEntropy with current weights
        proto_model = setup_proto_entropy_custom(
            base_model, 
            alpha_target=alpha_t, 
            alpha_separation=alpha_s, 
            alpha_coherence=alpha_c
        )
        
        # Evaluate
        acc = evaluate_model(proto_model, test_loader, description=f"ProtoEntropy ({name})")
        
        # Store result
        results.append({
            'name': name,
            'target': alpha_t,
            'separation': alpha_s,
            'coherence': alpha_c,
            'accuracy': acc
        })

        improvement = ""
        if baseline_acc:
            improvement = f" (Δ={((acc-baseline_acc)*100):+.2f}%)"
        
        print(f"    Accuracy: {acc*100:.2f}%{improvement}")

        if acc > best_acc:
            best_acc = acc
            best_config = config
            print(f"    *** NEW BEST! ***")

        # Clean up
        del proto_model
        del base_model
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    
    if baseline_acc:
        print(f"Baseline Accuracy (No Adaptation): {baseline_acc*100:.2f}%")
        print("-" * 80)
    
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    print(f"Best Configuration: {best_config['name']}")
    print(f"  Target={best_config['target']}, Separation={best_config['separation']}, Coherence={best_config['coherence']}")
    
    if baseline_acc:
        print(f"  Improvement over baseline: {(best_acc - baseline_acc)*100:+.2f}%")
    
    print("\n" + "="*80)
    print("All Results (sorted by accuracy):")
    print("="*80)
    print(f"{'Configuration':<30s} | {'T':>5s} | {'S':>5s} | {'C':>5s} | {'Accuracy':>10s} | {'vs Baseline':>12s}")
    print("-" * 80)
    
    # Sort by accuracy
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for res in sorted_results:
        improvement_str = ""
        if baseline_acc:
            improvement_str = f"{(res['accuracy'] - baseline_acc)*100:+6.2f}%"
        print(f"{res['name']:<30s} | {res['target']:5.2f} | {res['separation']:5.2f} | {res['coherence']:5.2f} | {res['accuracy']*100:9.2f}% | {improvement_str:>12s}")
    
    print("="*80)
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Find best without separation/coherence
    original = [r for r in results if r['separation'] == 0.0 and r['coherence'] == 0.0]
    if original:
        original_acc = original[0]['accuracy']
        print(f"Original ProtoEntropy (no Sep/Coh): {original_acc*100:.2f}%")
        if best_config['separation'] > 0 or best_config['coherence'] > 0:
            print(f"Best with Sep/Coh: {best_acc*100:.2f}% (Δ={((best_acc-original_acc)*100):+.2f}%)")
            if best_acc > original_acc:
                print("✓ Separation/Coherence losses IMPROVE performance")
            else:
                print("✗ Separation/Coherence losses DO NOT improve performance")
    
    # Find best with separation only
    sep_only = [r for r in results if r['separation'] > 0.0 and r['coherence'] == 0.0]
    if sep_only:
        best_sep = max(sep_only, key=lambda x: x['accuracy'])
        print(f"\nBest with Separation only: {best_sep['accuracy']*100:.2f}% (Sep={best_sep['separation']})")
    
    # Find best with coherence only
    coh_only = [r for r in results if r['separation'] == 0.0 and r['coherence'] > 0.0]
    if coh_only:
        best_coh = max(coh_only, key=lambda x: x['accuracy'])
        print(f"Best with Coherence only: {best_coh['accuracy']*100:.2f}% (Coh={best_coh['coherence']})")
    
    print("="*80)
    
    return results, best_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid Search for ProtoEntropy Loss Weights')
    
    default_model_path = './saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth'
    
    parser.add_argument('-model', type=str, default=default_model_path, help='Path to the saved model file')
    parser.add_argument('-gpuid', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-corruption', type=str, default='gaussian_noise', 
                       help='Type of corruption to apply')
    parser.add_argument('-severity', type=int, default=4, help='Severity of corruption (1-5)')
    parser.add_argument(
        '--on-the-fly',
        action='store_true',
        default=False,
        help='Force on-the-fly corruption generation (ignore pre-generated images).'
    )
    parser.add_argument(
        '--no-baseline',
        action='store_true',
        default=False,
        help='Skip baseline evaluation.'
    )
    
    args = parser.parse_args()
    
    use_pre_generated = not args.on_the_fly
    include_baseline = not args.no_baseline
    
    run_grid_search(args.model, args.gpuid, args.corruption, args.severity, 
                   use_pre_generated, include_baseline)

