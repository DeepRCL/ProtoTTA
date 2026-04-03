import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import train_and_test as tnt
import model # Necessary for torch.load to find the class definition
import push_greedy # Importing just in case the model object depends on it
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform

def run_inference(model_path, gpu_id='0', corruption=None, severity=1):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f'Using GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data loading
    print(f'Loading test data from: {test_dir}')
    if corruption:
        print(f'Applying corruption: {corruption} (Severity: {severity})')
    
    transform = get_corrupted_transform(img_size, mean, std, corruption, severity)
    
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    print(f'Test set size: {len(test_loader.dataset)}')

    # Load model
    print(f'Loading model from {model_path}')
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Model not found at {model_path}")

    # The models in this repo are saved as full objects (torch.save(model)), 
    # not state_dicts. So we load them directly.
    # Note: We need 'model' imported so pickle can find the class definition.
    # PyTorch 2.6+ defaults to weights_only=True, which breaks full model loading.
    # We disable it here since we trust the local checkpoint.
    ppnet = torch.load(model_path, weights_only=False)
    ppnet = ppnet.to(device)
    ppnet.eval()

    # Parameters for testing
    # class_specific is True by default in main.py
    class_specific = True 
    
    # Run inference
    print('Starting inference...')
    # We use the existing test function from train_and_test.py
    # it returns (accuracy, loss_dictionary)
    accu, test_loss_dict = tnt.test(model=ppnet, dataloader=test_loader,
                                    class_specific=class_specific, log=print, 
                                    clst_k=k, sum_cls=sum_cls)
    
    print('-' * 20)
    print(f'Inference Complete.')
    print(f'Final Accuracy: {accu*100:.2f}%')
    print('-' * 20)
    return accu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference for ProtoViT')
    
    # Default path to the best model based on your saved_models directory structure
    # You can change this to point to a specific push model like '.../4push0.8543.pth'
    default_model_path = './saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth'
    
    parser.add_argument('-model', type=str, default=default_model_path, help='Path to the saved model file')
    parser.add_argument('-gpuid', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-corruption', type=str, default=None, choices=['gaussian_noise'], help='Type of corruption to apply')
    parser.add_argument('-severity', type=int, default=1, help='Severity of corruption (1-5)')
    
    args = parser.parse_args()
    
    run_inference(args.model, args.gpuid, args.corruption, args.severity)

