import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import torch.optim as optim
import logging
import train_and_test as tnt
import model # Necessary for torch.load to find the class definition
import push_greedy # Importing just in case the model object depends on it
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform
import tent
import proto_entropy

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
    """Set up optimizer for tent adaptation.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError

def setup_tent(model):
    """Set up tent adaptation.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_proto_entropy(model):
    """Set up Prototype Entropy adaptation.
    """
    model = proto_entropy.configure_model(model)
    params, param_names = proto_entropy.collect_params(model)
    optimizer = setup_optimizer(params)
    # Use the ProtoEntropy wrapper instead of Tent
    proto_model = proto_entropy.ProtoEntropy(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return proto_model

def evaluate_model(model, loader, description="Inference"):
    """Helper to run evaluation loop."""
    print(f'\nStarting {description}...')
    class_specific = True 
    accu, test_loss_dict = tnt.test(model=model, dataloader=loader,
                                    class_specific=class_specific, log=print, 
                                    clst_k=k, sum_cls=sum_cls)
    print('-' * 20)
    print(f'{description} Complete.')
    print(f'Final Accuracy: {accu*100:.2f}%')
    print('-' * 20)
    return accu


def run_unified_inference(model_path, gpu_id='0', corruption=None, severity=1, mode='all'):
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
    else:
        print('Applying NO corruption (Clean Data)')
    
    transform = get_corrupted_transform(img_size, mean, std, corruption, severity)
    
    test_dataset = datasets.ImageFolder(test_dir, transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=8, pin_memory=False)

    print(f'Test set size: {len(test_loader.dataset)}')
    
    results = {}

    # --- NORMAL INFERENCE ---
    if mode in ['normal', 'all']:
        print(f'\n>>> Loading model for NORMAL inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        base_model.eval()

        acc = evaluate_model(base_model, test_loader, description="Normal Inference (No Adaptation)")
        results['Normal'] = acc
        
        # Clean up to free memory
        del base_model
        torch.cuda.empty_cache()

    # --- TENT INFERENCE ---
    if mode in ['tent', 'all']:
        print(f'\n>>> Loading model for TENT inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk (IMPORTANT: start from same state)
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        
        # Setup Tent (this switches mode to train for norm layers)
        print("Setting up Tent adaptation...")
        tent_model = setup_tent(base_model)
        
        acc = evaluate_model(tent_model, test_loader, description="Tent Adaptation Inference")
        results['Tent'] = acc
        
        # Clean up
        del tent_model
        torch.cuda.empty_cache()

    # --- PROTO ENTROPY INFERENCE ---
    if mode in ['proto', 'all']:
        print(f'\n>>> Loading model for PROTO ENTROPY inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        
        # Setup ProtoEntropy
        print("Setting up ProtoEntropy adaptation...")
        proto_model = setup_proto_entropy(base_model)
        
        acc = evaluate_model(proto_model, test_loader, description="ProtoEntropy Adaptation Inference")
        results['ProtoEntropy'] = acc
        
        # Clean up
        del proto_model
        torch.cuda.empty_cache()

    # --- FINAL SUMMARY ---
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset Corruption: {corruption if corruption else 'None'}")
    if corruption:
        print(f"Severity: {severity}")
    print("-" * 50)
    
    if 'Normal' in results:
        print(f"Normal       Accuracy: {results['Normal']*100:.2f}%")
    if 'Tent' in results:
        print(f"Tent         Accuracy: {results['Tent']*100:.2f}%")
    if 'ProtoEntropy' in results:
        print(f"ProtoEntropy Accuracy: {results['ProtoEntropy']*100:.2f}%")
    
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Inference for ProtoViT (Normal & Tent & ProtoEntropy)')
    
    default_model_path = './saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth'
    
    parser.add_argument('-model', type=str, default=default_model_path, help='Path to the saved model file')
    parser.add_argument('-gpuid', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-corruption', type=str, default='gaussian_noise', help='Type of corruption to apply')
    parser.add_argument('-severity', type=int, default=4, help='Severity of corruption (1-5)')
    parser.add_argument('-mode', type=str, default='all', choices=['normal', 'tent', 'loss', 'all'], help='Inference mode: normal, tent, proto, or all')
    
    args = parser.parse_args()
    
    run_unified_inference(args.model, args.gpuid, args.corruption, args.severity, args.mode)