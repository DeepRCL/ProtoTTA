#!/usr/bin/env python3
"""
Quick comparison script for ProtoEntropy variants.
Compares: Original, Importance-Weighted, and Confidence-Weighted.
"""

import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import train_and_test as tnt
import model
import proto_entropy
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform
from pathlib import Path
import torch.optim as optim

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
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params, lr=cfg.OPTIM.LR,
                         betas=(cfg.OPTIM.BETA, 0.999),
                         weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError

def setup_proto_entropy(model, use_importance=False, use_confidence=False):
    model = proto_entropy.configure_model(model)
    params, param_names = proto_entropy.collect_params(model)
    optimizer = setup_optimizer(params)
    
    proto_model = proto_entropy.ProtoEntropy(
        model, optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        use_prototype_importance=use_importance,
        use_confidence_weighting=use_confidence
    )
    
    return proto_model

def evaluate_model(model, loader):
    class_specific = True 
    accu, _ = tnt.test(model=model, dataloader=loader,
                      class_specific=class_specific, log=lambda x: None, 
                      clst_k=k, sum_cls=sum_cls)
    return accu

def main():
    parser = argparse.ArgumentParser(description='Compare ProtoEntropy Variants')
    parser.add_argument('-model', type=str, 
                       default='./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth')
    parser.add_argument('-gpuid', type=str, default='0')
    parser.add_argument('-corruption', type=str, default='gaussian_noise')
    parser.add_argument('-severity', type=int, default=4)
    parser.add_argument('--on-the-fly', action='store_true', default=False)
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    if args.corruption:
        corrupted_data_dir = Path('./datasets/cub200_c')
        corruption_path = corrupted_data_dir / args.corruption / str(args.severity)
        
        if not args.on_the_fly and corruption_path.exists():
            print(f'Using PRE-GENERATED: {corruption_path}')
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            test_dataset = datasets.ImageFolder(str(corruption_path), transform)
        else:
            print(f'Generating ON-THE-FLY: {args.corruption} (Severity: {args.severity})')
            transform = get_corrupted_transform(img_size, mean, std, args.corruption, args.severity)
            test_dataset = datasets.ImageFolder(test_dir, transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    
    print("\n" + "="*80)
    print("PROTOENTROPY VARIANTS COMPARISON")
    print("="*80)
    print(f"Corruption: {args.corruption} | Severity: {args.severity}")
    print(f"Test samples: {len(test_dataset)}")
    print("="*80)
    
    results = []
    
    # Variant 1: Original ProtoEntropy
    print("\n[1/4] Baseline (No Adaptation)...")
    base_model = torch.load(args.model, weights_only=False).to(device).eval()
    baseline_acc = evaluate_model(base_model, test_loader)
    results.append(('Baseline (No Adapt)', baseline_acc))
    print(f"      Accuracy: {baseline_acc*100:.2f}%")
    del base_model
    torch.cuda.empty_cache()
    
    # Variant 2: Original ProtoEntropy
    print("\n[2/4] ProtoEntropy (Original)...")
    base_model = torch.load(args.model, weights_only=False).to(device)
    proto_model = setup_proto_entropy(base_model, use_importance=False, use_confidence=False)
    original_acc = evaluate_model(proto_model, test_loader)
    results.append(('ProtoEntropy (Original)', original_acc))
    print(f"      Accuracy: {original_acc*100:.2f}% (Δ={((original_acc-baseline_acc)*100):+.2f}%)")
    del proto_model
    torch.cuda.empty_cache()
    
    # Variant 3: Importance-Weighted
    print("\n[3/4] ProtoEntropy (Importance-Weighted)...")
    base_model = torch.load(args.model, weights_only=False).to(device)
    proto_model = setup_proto_entropy(base_model, use_importance=True, use_confidence=False)
    importance_acc = evaluate_model(proto_model, test_loader)
    results.append(('ProtoEntropy (Importance)', importance_acc))
    print(f"      Accuracy: {importance_acc*100:.2f}% (Δ={((importance_acc-baseline_acc)*100):+.2f}%)")
    del proto_model
    torch.cuda.empty_cache()
    
    # Variant 4: Confidence-Weighted
    print("\n[4/4] ProtoEntropy (Confidence-Weighted)...")
    base_model = torch.load(args.model, weights_only=False).to(device)
    proto_model = setup_proto_entropy(base_model, use_importance=False, use_confidence=True)
    confidence_acc = evaluate_model(proto_model, test_loader)
    results.append(('ProtoEntropy (Confidence)', confidence_acc))
    print(f"      Accuracy: {confidence_acc*100:.2f}% (Δ={((confidence_acc-baseline_acc)*100):+.2f}%)")
    del proto_model
    torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<30s} | {'Accuracy':>10s} | {'vs Baseline':>12s} | {'vs Original':>12s}")
    print("-" * 80)
    
    for name, acc in results:
        vs_baseline = f"{(acc - baseline_acc)*100:+.2f}%" if name != 'Baseline (No Adapt)' else "---"
        vs_original = f"{(acc - original_acc)*100:+.2f}%" if name not in ['Baseline (No Adapt)', 'ProtoEntropy (Original)'] else "---"
        print(f"{name:<30s} | {acc*100:9.2f}% | {vs_baseline:>12s} | {vs_original:>12s}")
    
    print("="*80)
    
    # Winner
    best_idx = max(range(1, len(results)), key=lambda i: results[i][1])
    best_name, best_acc = results[best_idx]
    
    print(f"\n🏆 BEST METHOD: {best_name}")
    print(f"   Accuracy: {best_acc*100:.2f}%")
    print(f"   Improvement over baseline: {(best_acc - baseline_acc)*100:+.2f}%")
    print(f"   Improvement over original: {(best_acc - original_acc)*100:+.2f}%")
    print("="*80)

if __name__ == '__main__':
    main()

