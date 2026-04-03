import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import train_and_test as tnt
import model # Necessary for torch.load to find the class definition
import push_greedy # Importing just in case the model object depends on it
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform
import tent
import loss_adapt
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

def setup_loss_adapt(model, sparsity_weight, clustering_weight):
    """Set up Loss-based adaptation with specific weights."""
    model = loss_adapt.configure_model(model)
    params, param_names = loss_adapt.collect_params(model)
    optimizer = setup_optimizer(params)
    loss_model = loss_adapt.LossAdapt(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        sparsity_weight=sparsity_weight,
        clustering_weight=clustering_weight
    )
    return loss_model

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

def run_grid_search(model_path, gpu_id='0', corruption='gaussian_noise', severity=3, use_pre_generated=True):
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
            # Load pre-generated corrupted dataset (corruption already applied)
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
        num_workers=8, pin_memory=False)

    print(f'Test set size: {len(test_loader.dataset)}')
    
    # Define grid
    sparsity_weights = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    clustering_weights = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    results = []
    best_acc = -1.0
    best_config = None

    print(f"\nRunning Grid Search for Corruption: {corruption} (Severity: {severity})")
    print(f"Sparsity Weights: {sparsity_weights}")
    print(f"Clustering Weights: {clustering_weights}")
    print("="*60)

    for sw in sparsity_weights:
        for cw in clustering_weights:
            print(f"\nTesting Configuration: Sparsity={sw}, Clustering={cw}")
            
            # Load fresh model for each iteration to avoid history contamination
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model not found at {model_path}")
                 
            base_model = torch.load(model_path, weights_only=False)
            base_model = base_model.to(device)
            
            # Setup LossAdapt with current weights
            loss_model = setup_loss_adapt(base_model, sparsity_weight=sw, clustering_weight=cw)
            
            # Evaluate
            acc = evaluate_model(loss_model, test_loader, description=f"LossAdapt (S={sw}, C={cw})")
            
            # Store result
            results.append({
                'sparsity': sw,
                'clustering': cw,
                'accuracy': acc
            })

            if acc > best_acc:
                best_acc = acc
                best_config = {'sparsity': sw, 'clustering': cw}
                print(f"*** New Best Accuracy: {best_acc*100:.2f}% (S={sw}, C={cw}) ***")

            # Clean up
            del loss_model
            del base_model
            torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("GRID SEARCH COMPLETE")
    print("="*60)
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    print(f"Best Configuration: Sparsity={best_config['sparsity']}, Clustering={best_config['clustering']}")
    print("-" * 60)
    print("All Results:")
    print(f"{'Sparsity':<10} | {'Clustering':<10} | {'Accuracy':<10}")
    for res in results:
        print(f"{res['sparsity']:<10.2f} | {res['clustering']:<10.2f} | {res['accuracy']*100:.2f}%")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grid Search for LossAdapt Weights')
    
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
    
    args = parser.parse_args()
    
    use_pre_generated = not args.on_the_fly
    
    run_grid_search(args.model, args.gpuid, args.corruption, args.severity, use_pre_generated)

