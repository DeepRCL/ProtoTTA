#!/usr/bin/env python3
"""
Simple test script to verify MEMO implementation works correctly.

Usage:
    python test_memo.py --model ./saved_models/best_model.pth
"""

import os
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import model  # Necessary for torch.load
from settings import img_size
from preprocess import mean, std
import memo_adapt


def main():
    parser = argparse.ArgumentParser(description='Test MEMO implementation')
    parser.add_argument('--model', type=str, 
                       default='./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth',
                       help='Path to saved model')
    parser.add_argument('--data_dir', type=str,
                       default='./datasets/cub200_cropped/test_cropped/',
                       help='Path to test data')
    parser.add_argument('--gpuid', type=str, default='0',
                       help='GPU ID to use')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test')
    args = parser.parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'\nLoading model from {args.model}...')
    base_model = torch.load(args.model, weights_only=False)
    base_model = base_model.to(device)
    base_model.eval()
    
    # Setup MEMO
    print('Setting up MEMO...')
    memo_model = memo_adapt.setup_memo(
        base_model,
        lr=0.00025,
        batch_size=32,  # Number of augmented views
        steps=1         # Number of adaptation steps
    )
    
    # Load test data
    print(f'\nLoading test data from {args.data_dir}...')
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dataset = datasets.ImageFolder(args.data_dir, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # MEMO processes one image at a time
        shuffle=True,
        num_workers=2
    )
    
    # Test MEMO on a few samples
    print(f'\nTesting MEMO on {args.num_samples} samples...')
    print('='*60)
    
    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(loader):
        if i >= args.num_samples:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass with MEMO adaptation
        with torch.no_grad():
            outputs, min_distances, values = memo_model(images)
        
        # Get prediction
        _, predicted = outputs.max(1)
        
        # Check correctness
        is_correct = (predicted == labels).item()
        correct += is_correct
        total += 1
        
        print(f'Sample {i+1}/{args.num_samples}: '
              f'Predicted: {predicted.item()}, '
              f'True: {labels.item()}, '
              f'Correct: {is_correct}')
    
    print('='*60)
    print(f'\nAccuracy: {correct}/{total} = {100*correct/total:.2f}%')
    print('\nMEMO test completed successfully!')


if __name__ == '__main__':
    main()

