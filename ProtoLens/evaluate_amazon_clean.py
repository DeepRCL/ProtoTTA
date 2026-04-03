
import os
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import AutoTokenizer

# Import ProtoLens modules
from PLens import BERTClassifier
from utils import TextClassificationDataset

def load_model_standalone(model_path, device):
    """Load trained ProtoLens model."""
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    saved_args = checkpoint.get('pnfrl_args', {})
    
    class Args:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    args = Args(
        bert_model_name=saved_args.get('bert_model_name', 'all-mpnet-base-v2'),
        num_classes=saved_args.get('num_classes', 2),
        prototype_num=saved_args.get('prototype_num', 50),
        batch_size=saved_args.get('batch_size', 32),
        hidden_dim=saved_args.get('hidden_dim', 768),
        max_length=saved_args.get('max_length', 512),
        data_set='Yelp', base_folder='Datasets', gaussian_num=6, window_size=5
    )
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    
    model = BERTClassifier(
        args=args, bert_model_name='sentence-transformers/all-mpnet-base-v2',
        num_classes=args.num_classes, num_prototype=args.prototype_num,
        batch_size=args.batch_size, hidden_dim=args.hidden_dim,
        max_length=args.max_length, tokenizer=tokenizer
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, tokenizer, args

def load_clean_data_standalone(data_dir):
    """Load clean (uncorrupted) Amazon test dataset."""
    # Priority 1: Parent Amazon directory
    amazon_dir = os.path.join(os.path.dirname(data_dir.rstrip('/')), 'Amazon')
    test_path = os.path.join(amazon_dir, 'test.csv')
    if os.path.exists(test_path):
        print(f"Found clean data at {test_path}")
        return pd.read_csv(test_path)
    
    # Priority 2: Common clean data filenames in current data_dir
    for clean_name in ['amazon_c_clean.csv', 'amazon_test.csv', 'amazon_clean.csv', 'test.csv']:
        filepath = os.path.join(data_dir, clean_name)
        if os.path.exists(filepath):
            print(f"Found clean data at {filepath}")
            return pd.read_csv(filepath)
    
    raise FileNotFoundError(f"No clean data found in {amazon_dir} or {data_dir}")

def evaluate_clean():
    # Configuration
    model_path = 'log_folder/Yelp/_Yelp_fine-tune_all-mpnet-base-v2_gNum_6_ws_5_e_15_pNum_50_lr0.0005/model.pth'
    data_dir = 'Datasets/Amazon-C'
    batch_size = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer, model_args = load_model_standalone(model_path, device)
    
    # Load data
    try:
        df = load_clean_data_standalone(data_dir)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Evaluating on {len(df)} samples...")
    
    # Create dataloader
    texts = df['review'].tolist()
    labels = df['sentiment'].tolist()
    dataset = TextClassificationDataset(texts, labels, tokenizer, model_args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)
            labels_batch = batch['label'].to(device)
            
            result = model(
                input_ids=input_ids, attention_mask=attention_mask,
                special_tokens_mask=special_tokens_mask, mode="test",
                original_text=batch['original_text'], current_batch_num=0
            )
            
            outputs = result[0] if isinstance(result, tuple) else result
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels_batch.cpu().tolist())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print("\n" + "="*40)
    print(f"Performance on Clean Amazon Dataset")
    print("="*40)
    print(f"Model Path: {model_path}")
    print(f"Total Samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("-" * 40)
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    print("="*40)

if __name__ == '__main__':
    evaluate_clean()
