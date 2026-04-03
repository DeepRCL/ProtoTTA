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

def setup_proto_entropy(model, use_importance=False):
    """Set up ProtoEntropy adaptation."""
    model = proto_entropy.configure_model(model)
    params, param_names = proto_entropy.collect_params(model)
    optimizer = setup_optimizer(params)
    
    # Original ProtoEntropy: no separation/coherence, just target entropy
    proto_model = proto_entropy.ProtoEntropy(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        alpha_target=1.0,
        alpha_separation=0.0,
        alpha_coherence=0.0,
        use_prototype_importance=use_importance
    )
    
    return proto_model

def evaluate_model(model, loader, description="Inference"):
    """Helper to run evaluation loop."""
    print(f'{description}...')
    class_specific = True 
    accu, test_loss_dict = tnt.test(model=model, dataloader=loader,
                                    class_specific=class_specific, log=lambda x: None, 
                                    clst_k=k, sum_cls=sum_cls)
    return accu

def main(model_path, gpu_id='0', corruption='gaussian_noise', severity=4, 
         use_pre_generated=True):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f'Using GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data loading
    if corruption:
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
    print("")
    print("="*80)
    print("PROTOTYPE IMPORTANCE WEIGHTING TEST")
    print("="*80)
    print(f"Corruption: {corruption} | Severity: {severity}")
    print("="*80)
    
    results = {}
    
    # Test 1: Baseline (No Adaptation)
    print("\n[1/3] Evaluating Baseline (No Adaptation)")
    print("-" * 80)
    base_model = torch.load(model_path, weights_only=False)
    base_model = base_model.to(device)
    base_model.eval()
    baseline_acc = evaluate_model(base_model, test_loader, description="Baseline")
    results['baseline'] = baseline_acc
    print(f"Accuracy: {baseline_acc*100:.2f}%")
    del base_model
    torch.cuda.empty_cache()
    
    # Test 2: Original ProtoEntropy (uniform weighting)
    print("\n[2/3] Evaluating Original ProtoEntropy (Uniform Weighting)")
    print("-" * 80)
    base_model = torch.load(model_path, weights_only=False)
    base_model = base_model.to(device)
    proto_model = setup_proto_entropy(base_model, use_importance=False)
    original_acc = evaluate_model(proto_model, test_loader, 
                                  description="ProtoEntropy (Original)")
    results['original'] = original_acc
    improvement = (original_acc - baseline_acc) * 100
    print(f"Accuracy: {original_acc*100:.2f}% (Δ={improvement:+.2f}%)")
    del proto_model
    del base_model
    torch.cuda.empty_cache()
    
    # Test 3: ProtoEntropy with Prototype Importance Weighting
    print("\n[3/3] Evaluating ProtoEntropy with Prototype Importance Weighting")
    print("-" * 80)
    base_model = torch.load(model_path, weights_only=False)
    base_model = base_model.to(device)
    proto_model = setup_proto_entropy(base_model, use_importance=True)
    importance_acc = evaluate_model(proto_model, test_loader, 
                                   description="ProtoEntropy (Importance-Weighted)")
    results['importance'] = importance_acc
    improvement = (importance_acc - baseline_acc) * 100
    print(f"Accuracy: {importance_acc*100:.2f}% (Δ={improvement:+.2f}%)")
    del proto_model
    del base_model
    torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<40s} | {'Accuracy':>10s} | {'vs Baseline':>12s}")
    print("-" * 80)
    print(f"{'Baseline (No Adaptation)':<40s} | {results['baseline']*100:9.2f}% | {'---':>12s}")
    print(f"{'ProtoEntropy (Original - Uniform)':<40s} | {results['original']*100:9.2f}% | {(results['original']-results['baseline'])*100:+11.2f}%")
    print(f"{'ProtoEntropy (Importance-Weighted)':<40s} | {results['importance']*100:9.2f}% | {(results['importance']-results['baseline'])*100:+11.2f}%")
    print("="*80)
    
    # Comparison
    diff = (results['importance'] - results['original']) * 100
    print("\nDirect Comparison:")
    print(f"Importance-Weighted vs Original: {diff:+.2f}%")
    
    if results['importance'] > results['original']:
        print("✓ Prototype Importance Weighting IMPROVES performance!")
    elif results['importance'] < results['original']:
        print("✗ Prototype Importance Weighting DECREASES performance")
    else:
        print("= Prototype Importance Weighting has NO EFFECT")
    
    print("="*80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test Prototype Importance Weighting for ProtoEntropy'
    )
    
    default_model_path = './saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth'
    
    parser.add_argument('-model', type=str, default=default_model_path, 
                       help='Path to the saved model file')
    parser.add_argument('-gpuid', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-corruption', type=str, default='gaussian_noise', 
                       help='Type of corruption to apply')
    parser.add_argument('-severity', type=int, default=4, 
                       help='Severity of corruption (1-5)')
    parser.add_argument('--on-the-fly', action='store_true', default=False,
                       help='Force on-the-fly corruption generation')
    
    args = parser.parse_args()
    
    use_pre_generated = not args.on_the_fly
    
    main(args.model, args.gpuid, args.corruption, args.severity, use_pre_generated)

