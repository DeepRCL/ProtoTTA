
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import BertConfig, BertModel
import os
import sys
import warnings
# Suppress scipy/torch.compile warnings that flood output
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')
warnings.filterwarnings('ignore', message='.*cudagraph partition.*')
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import torch.nn.functional as F
from torch import where, rand_like, zeros_like, log, sigmoid
from torch.nn import Module
# from settings import *
# # import torch.cuda.amp.GradScaler
# from settings import *
import numpy as np
from transformers import AutoTokenizer, BertModel
from transformers import  DistilBertModel, DistilBertTokenizerFast
import logging
from scipy.spatial.distance import cdist
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from sklearn.model_selection import KFold
from utils import *
# from model_ours import *

from PLens import *
import gc
# from gpu_mem_track import MemTracker  # Not used, commented out
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import os 

import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from Gumbel_Sigmoid import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def gold_eval(model, data_loader, p_sent2pid, device, epo_num):
    # NOTE: This function is not used in training - commenting out hardcoded path
    # sentence_pool = pd.read_csv("/home/bwei2/ProtoTextClassification/Data/IMDB_cluster_" + str(self.num_prototypes) + "_to_sub_sentence.csv", index_col=0, header=None).to_numpy()
    model.eval()
    predictions = []
    actual_labels = []
    document = docx.Document()
    # with torch.no_grad():
    for batch in data_loader:
        original_text = batch['original_text']
        input_ids = batch['input_ids'].to(device)
        input_attention_mask = batch['attention_mask'].to(device)
        special_tokens_mask = batch['special_tokens_mask'].to(device)
        label = batch['label_text']
        pid = [p_sent2pid[i] for i in label]
        attention_mask_label = batch["attention_mask_label"].to(device)
        special_tokens_mask_label = batch["special_tokens_mask_label"].to(device)
        input_ids_label = batch["input_ids_label"].to(device)
        # logging.info(special_tokens_mask_label)
        label_ids = (1 - special_tokens_mask_label) * input_ids_label
        # Initialize the label mask with zeros
        label_mask = torch.zeros_like(input_ids)
        # non_zero_elements = label_ids[label_ids != 0]
        seq = []
        for i in range(label_ids.shape[0]):
            # Get non-zero elements from label_ids for this sequence
            non_zero_elements = label_ids[i][label_ids[i] != 0]
            seq.append(non_zero_elements)
            # Check if the non-zero elements exist in input_ids
            for elem in non_zero_elements:
                # Get the index where the element appears in input_ids
                indices = torch.where(input_ids[i] == elem)[0]
                # Set these positions in the label mask to 1
                label_mask[i][indices] = 1
        # label_mask = find_and_replace_subsequence(label_mask, seq)

        # words, _, df, vocab = model.get_words(original_text)
        # label_words_in_order = []
        # for label_text in label:
        #     _, temp, _, label_vocab = model.get_words([label_text])
        #     label_words_in_order.append(temp)

        # token_embeddings, candidates_id, candidates = model.get_token_embedding(original_text, words, vocab)
        # label_mask = np.full((len(original_text), model.max_length), 0)
        # for index in range(len(original_text)):
        #     for i,_ in enumerate(candidates[index]):
        #         if i >= model.max_length:
        #             continue
        #         if candidates[index][i] in label_words_in_order[index]:
        #             label_mask[index, i] = 1
        # label_mask = torch.tensor(label_mask).cuda()
        # attention_mask = torch.tensor(np.where(candidates_id == 'None', 0, 1)).cuda() # candidates = torch.tensor(candidates)
        outputs = model.bert(input_ids=input_ids, attention_mask=input_attention_mask, output_hidden_states=True)
        input_token_emb = F.normalize(outputs[0], p=2, dim=-1) # (16, 512, 768)
        align = F.normalize(model.bert(input_ids_label, attention_mask_label).last_hidden_state, p=2, dim=-1)
        attention_logits, all_selected_token_index = model.attention_layer(input_token_emb, align, att_mask_emb=special_tokens_mask, att_mask_proto=attention_mask_label)
        mask = model.AdaptiveMask(attention_logits, all_selected_token_index)  
        selected_mask = mask * input_attention_mask.unsqueeze(-1)

        # p_vecs = model.bert.encode(label, normalize_embeddings=True, convert_to_tensor=True)
        # attention_logits = token_embeddings @ p_vecs.T
        # mask = model.AdaptiveMask(attention_logits)[torch.arange(len(original_text)), :, torch.arange(len(original_text))] #(16, 512)
        # selected_mask = (mask * attention_mask).cpu().detach().numpy().astype(int) #(mask * attention_mask).cpu().detach().numpy().astype(int)
        # selected_mask_ = mask * attention_mask
        # label_mask_ = torch.tensor(label_mask).cuda()
        # proto_acc = torch.sum(selected_mask * label_mask, dim=1) / torch.sum(label_mask, dim=1)
        # proto_acc_mean = torch.mean(proto_acc)
        # proto_recall = torch.sum(selected_mask * label_mask, dim=1) / torch.sum(selected_mask, dim=1)
        # proto_recall_mean = torch.mean(proto_recall)
        # logging.info("acc:", proto_acc_mean)
        # logging.info("recall:", proto_recall_mean)
    #     word_array = np.where(selected_mask == 1, candidates_id, "None")
    #     for i in range(len(original_text)):
    #         selected_words = []
    #         for id_ in word_array[i]:
    #             if id_ != "None":
    #                 selected_words.append(words[int(id_)])
    #         positions = find_keywords(selected_words, original_text[i])
    #         document = highlight_sections_docx(positions, original_text[i], document, label[i])
    #     # selected_sent = [
    #     #         [i for i in word_array[idx, :] if i is not None]
    #     #         for idx in range(len(original_text))
    #     #     ]
    # document.save("/home/bwei2/ProtoTextClassification/_test/" + str(epo_num) + ".docx")
    # return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    incorrect_indices = []
    with torch.no_grad():
        for batch_num,batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            special_tokens_mask = batch['special_tokens_mask'].to(device)
            labels = batch['label'].to(device)
            offset_mapping = batch['offset_mapping']
            processed_text = batch['processed_text']
            original_text = batch['original_text']
            result = model(input_ids=input_ids, attention_mask=attention_mask,special_tokens_mask=special_tokens_mask, mode="test", original_text = original_text, current_batch_num=batch_num)
            outputs = result[0]  # Extract logits from (logits, loss_mu, augmented_loss, similarity)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    for idx, (pred, actual) in enumerate(zip(predictions, actual_labels)):
        if pred != actual:
            incorrect_indices.append(idx)
    logging.info(incorrect_indices)
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def find_closet_test_sentence(model, data_loader, device):
    # NOTE: This function is not used in training - commenting out hardcoded path
    model.eval()
    # df = pd.read_csv("/home/bwei2/ProtoTextClassification/Data/test_imdb.csv")
    # test_texts = df['review'].tolist()
    return {}
    test_ins_emb = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:

            original_text = batch['original_text']
            test_ins_emb.append(model.bert.encode(original_text, normalize_embeddings=True, convert_to_tensor=False))
    all_test_emb = np.stack(test_ins_emb, axis=0)
    distances = cosine_similarity(model.prototype_vectors, all_test_emb)
    closest_sentences = {}
    top_k = 5
    for proto_idx, proto_distances in enumerate(distances):
        # Get indices of the top_k most similar sentences
        top_k_indices = np.argsort(proto_distances)[-top_k:][::-1]
        # Retrieve the actual sentences
        closest_sentences[proto_idx] = [test_texts[idx] for idx in top_k_indices]
    
    return closest_sentences
    



def train_step(model, val_dataloader, data_loader, optimizer, scheduler, device, new_proto, tau=1, train_texts=None, epoch=0):
    from tqdm import tqdm
    
    model.train()
    total_loss = 0
    model.train_text = []
    
    # Use tqdm for progress bar - use file=sys.stderr to avoid interference with warnings
    # mininterval=0.1 ensures updates happen frequently
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}", 
                file=sys.stderr, mininterval=0.1, maxinterval=1.0)
    
    for (batch_num, batch) in pbar:
        model.batch_num = batch_num
        LOG = False
        optimizer.zero_grad()
        # Use non_blocking transfers for faster data loading (overlaps with GPU compute)
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        special_tokens_mask = batch['special_tokens_mask'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        offset_mapping = batch['offset_mapping'].to(device, non_blocking=True)
        processed_text = batch['processed_text']
        original_text = batch['original_text']
        # model.train_text.extend(original_text)
        # Use mixed precision training for speedup
        with autocast(device_type='cuda'):
            result = model(input_ids=input_ids, attention_mask=attention_mask, special_tokens_mask=special_tokens_mask, 
                            new_proto=new_proto, log=LOG, tau=tau, offset_mapping=offset_mapping, processed_text=processed_text, 
                            current_batch_num=batch_num, original_text = original_text)
            outputs, loss_mu, augmented_loss = result[0], result[1], result[2]  # Unpack first 3 values
            # span_loss = model.AdaptiveMask.get_loss()
            loss = nn.CrossEntropyLoss()(outputs, labels) + 0.1 * loss_mu - 0.001 * model.diversity_loss
        
        total_loss += loss.item()
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/(batch_num+1):.4f}'})
        
        # Use GradScaler for mixed precision
        if not hasattr(train_step, '_scaler'):
            train_step._scaler = torch.cuda.amp.GradScaler()
        
        train_step._scaler.scale(loss).backward()
        train_step._scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        train_step._scaler.step(optimizer)
        train_step._scaler.update()
        scheduler.step()
    
    pbar.close()
    return total_loss / len(data_loader)

def load_model(path, device_id, log_file=None, distributed=True):
    checkpoint = torch.load(path, map_location='cpu')
    saved_args = checkpoint['pnfrl_args']
    pnfrl = BERTClassifier(bert_model_name=saved_args['bert_model_name'], 
                           num_classes=saved_args['num_classes'], 
                           num_prototype=saved_args['prototype_num'], 
                           batch_size=saved_args['batch_size'], 
                           hidden_dim=saved_args['hidden_dim'], 
                           max_length=saved_args['max_length'])
    pnfrl.load_state_dict(checkpoint['pnfrl_args'])
    return pnfrl

def train_model(gpu, args):
    # Speed optimizations for maximum performance (memory not a concern)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:250,expandable_segments:True"
    # Suppress torch.compile warnings about CPU operations (scipy/CountVectorizer)
    os.environ["TORCHDYNAMO_VERBOSE"] = "0"
    
    rank = args.nr * args.gpus + gpu
    torch.manual_seed(args.r_seed)
    device_id = args.device_ids[gpu]
    torch.cuda.set_device(device_id)
    
    # Enable cuDNN benchmark for consistent input sizes (significant speedup)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    # Enable TensorFloat-32 for faster training on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log_file = args.log
    if gpu == 0:
        writer = SummaryWriter(args.folder_path)
        is_rank0 = True
    else:
        writer = None
        is_rank0 = False
    if is_rank0:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
        if log_file is None:
            logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
        else:
            logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
    adam_epsilon = 1e-8
    num_epochs = args.epoch
    dataset = args.data_set
    train_dataloader, val_dataloader, tokenizer, train_texts = get_data_loader(dataset,args.dataset_path, args.world_size, rank, args.batch_size, args.max_length, args.bert_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_warmup_steps = 0
    num_training_steps = len(train_dataloader)*num_epochs
    model = BERTClassifier(args=args, bert_model_name=args.bert_model_name, num_classes=args.num_classes, 
                           num_prototype=args.prototype_num, 
                           batch_size=args.batch_size, hidden_dim=args.hidden_dim, max_length=args.max_length, 
                           tokenizer=tokenizer).to(device)
    model.tokenizer = tokenizer
    model.args = args
    
    # Skip torch.compile() - it causes issues with scipy/CountVectorizer CPU operations
    # The warnings flood output and slow down the first forward pass significantly
    # Other optimizations (mixed precision, cuDNN benchmark, etc.) still provide good speedup
    # Uncomment below if you want to try compilation (may work better in future PyTorch versions)
    # try:
    #     if hasattr(torch, 'compile'):
    #         print("🚀 Compiling model with torch.compile()...")
    #         model = torch.compile(model, mode='reduce-overhead', fullgraph=False)
    #         print("✓ Model compiled successfully!")
    # except Exception as e:
    #     print(f"⚠️  torch.compile() failed: {e}")
    print("ℹ️  Skipping torch.compile() due to scipy/CountVectorizer compatibility issues")
    print("   Other speed optimizations (mixed precision, cuDNN, etc.) are still active")
    best_result = 0
    specific_param_left = model.AdaptiveMask.current_val_left
    specific_param_right =  model.AdaptiveMask.current_val_right
    other_params = [param for name, param in model.named_parameters() if param is not specific_param_left and param is not specific_param_right]
    # optimizer = Adam([
    #     {'params': specific_param_left, 'lr': 0.001},  # Learning rate for layer1
    #     {'params': specific_param_right, 'lr': 0.001},
    #     {'params': other_params}, 
    #                    ], lr=args.learning_rate, eps=adam_epsilon)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=adam_epsilon, weight_decay=1e-5 )
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    new_proto = None
    Total_Loss = []
    tau = 0.6
    
    print("\n" + "="*80)
    print(f"Starting Training: {num_epochs} epochs")
    print("="*80 + "\n")
    
    for epoch in range(num_epochs):
        # Removed torch.cuda.empty_cache() - not needed with abundant memory
        
        # Training
        avg_loss = train_step(model, val_dataloader, train_dataloader, optimizer, scheduler, device, 
                             new_proto, tau=tau, train_texts=train_texts, epoch=epoch)
        Total_Loss.append(avg_loss)
        
        # Validation
        print(f"\nEvaluating Epoch {epoch + 1}...")
        accuracy, report = evaluate(model, val_dataloader, device)
        
        # Save model info
        pnfrl_args = {'bert_model_name': args.bert_model_name, 'num_classes': args.num_classes, 
                    'prototype_num': args.prototype_num, 'batch_size': args.batch_size, 
                    'hidden_dim': args.hidden_dim, 'max_length': args.max_length}
        
        # Save best model
        if accuracy > best_result: 
            best_result = accuracy
            torch.save({'model_state_dict': model.state_dict(), 'pnfrl_args': pnfrl_args}, args.model_path)
            print(f"✓ New best model saved! Validation Accuracy: {accuracy:.4f}")
        
        # Print epoch summary
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{num_epochs} Summary")
        print(f"{'='*80}")
        print(f"Average Training Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Best Validation Accuracy: {best_result:.4f}")
        print(f"{'='*80}\n")
        
        # Log to file
        logging.info(f"\n{'='*80}")
        logging.info(f"Epoch {epoch + 1}/{num_epochs} Complete")
        logging.info(f"Average Training Loss: {avg_loss:.4f}")
        logging.info(f"Validation Accuracy: {accuracy:.4f}")
        logging.info(f"Best Validation Accuracy: {best_result:.4f}")
        logging.info(f"Classification Report:\n{report}")
        logging.info(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_result:.4f}")
    print(f"Model saved to: {args.model_path}")
    print(f"{'='*80}\n")
    
    # Load best model and evaluate on test set
    if is_rank0:
        print(f"\n{'='*80}")
        print("Evaluating Best Model on Test Set")
        print(f"{'='*80}\n")
        
        # Load the best saved model
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Loaded best model from checkpoint")
        
        # Load test data
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Datasets")
        test_file = os.path.join(base, dataset, "test.csv")
        
        if os.path.exists(test_file):
            print(f"Loading test set from: {test_file}")
            test_texts, test_labels = load_data(test_file)
            print(f"✓ Loaded {len(test_texts)} test samples")
            
            # Create test dataloader
            test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, args.max_length)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
            
            # Evaluate on test set
            print("\nRunning test evaluation...")
            test_accuracy, test_report = evaluate(model, test_loader, device)
            
            # Print test results
            print(f"\n{'='*80}")
            print(f"FINAL TEST RESULTS on {dataset}")
            print(f"{'='*80}")
            print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"Best Validation Accuracy: {best_result:.4f} ({best_result*100:.2f}%)")
            print(f"\nTest Classification Report:")
            print(test_report)
            print(f"{'='*80}\n")
            
            # Log test results
            logging.info(f"\n{'='*80}")
            logging.info(f"FINAL TEST RESULTS on {dataset}")
            logging.info(f"{'='*80}")
            logging.info(f"Test Accuracy: {test_accuracy:.4f}")
            logging.info(f"Best Validation Accuracy: {best_result:.4f}")
            logging.info(f"Test Classification Report:\n{test_report}")
            logging.info(f"{'='*80}\n")
            
            # Save test results to file
            results_file = os.path.join(args.folder_path, 'test_results.txt')
            with open(results_file, 'w') as f:
                f.write(f"Test Set Evaluation Results\n")
                f.write(f"{'='*80}\n")
                f.write(f"Dataset: {dataset}\n")
                f.write(f"Model: {args.model_path}\n")
                f.write(f"Best Validation Accuracy: {best_result:.4f}\n")
                f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
                f.write("Test Classification Report:\n")
                f.write(test_report)
            print(f"✓ Test results saved to: {results_file}")
        else:
            print(f"⚠️  Test file not found: {test_file}")
            print("   Skipping test evaluation")



def train_main(args):
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    # mp.spawn(train_model, nprocs=args.gpus, args=(args,),join=True)
    train_model(0, args)


if __name__ == '__main__':
    from args import pnfrl_args
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for arg in vars(pnfrl_args):
        print(arg, getattr(pnfrl_args, arg))
    train_main(pnfrl_args)
    # test_model(pnfrl_args)