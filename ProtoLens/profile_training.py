"""
Profile training to find bottlenecks
"""
import torch
import time
from utils import get_data_loader
from PLens import BERTClassifier
from args import pnfrl_args

# Setup
device = torch.device("cuda")
args = pnfrl_args
args.data_set = 'Yelp'
args.prototype_num = 50

# Load one batch
train_dataloader, val_dataloader, tokenizer, train_texts = get_data_loader(
    'Yelp', args.dataset_path, 1, 0, 16, 512, 'sentence-transformers/all-mpnet-base-v2'
)

# Initialize model
model = BERTClassifier(
    args=args,
    bert_model_name='sentence-transformers/all-mpnet-base-v2',
    num_classes=2,
    num_prototype=50,
    batch_size=16,
    hidden_dim=768,
    max_length=512,
    tokenizer=tokenizer
).to(device)

model.train()

# Get one batch
batch = next(iter(train_dataloader))

# Warm up
print("Warming up...")
with torch.no_grad():
    for _ in range(3):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        original_text = batch['original_text']
        _ = model(input_ids=input_ids, attention_mask=attention_mask, 
                 special_tokens_mask=special_tokens_mask, original_text=original_text, 
                 current_batch_num=0)

# Profile
print("\nProfiling forward pass...")
times = []
for i in range(10):
    torch.cuda.synchronize()
    start = time.time()
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    special_tokens_mask = batch['special_tokens_mask'].to(device)
    original_text = batch['original_text']
    
    result = model(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
        original_text=original_text,
        current_batch_num=i
    )
    outputs, loss_mu, augmented_loss = result[0], result[1], result[2]
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Iteration {i+1}: {elapsed:.3f}s")

print(f"\nAverage time per forward pass: {sum(times)/len(times):.3f}s")
print(f"Expected batches/second: {1/(sum(times)/len(times)):.2f}")
print(f"Expected time per epoch (23827 batches): {23827 * (sum(times)/len(times)) / 60:.1f} minutes")
