"""
Generate prototype initialization files for ProtoLens.

This script:
1. Loads training data
2. Extracts sub-sentences from reviews
3. Computes sentence embeddings using SentenceTransformer
4. Performs K-means clustering to get prototype centers
5. Saves cluster centers and representative sentences

Based on ProtoLens paper methodology.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import regex as re
from tqdm import tqdm
import torch
import pickle
import warnings
warnings.filterwarnings('ignore')

def process_sentence(text):
    """Process text for sentence splitting"""
    text = text.lower()
    text = " " + text + "  "
    text = text.replace("\n", " ")
    return text

def extract_sub_sentences(texts, window_size=5, max_samples=None):
    """
    Extract sub-sentences (n-grams) from texts.
    Similar to ProtoLens approach using sliding window.
    """
    print(f"\nExtracting sub-sentences with window size {window_size}...")
    
    if max_samples:
        texts = texts[:max_samples]
    
    # Use CountVectorizer to extract n-grams
    vectorizer = CountVectorizer(
        ngram_range=(window_size, window_size),
        stop_words=None,
        max_features=50000  # Limit vocabulary size
    )
    
    # Fit and get n-grams
    print("Fitting vectorizer...")
    vectorizer.fit(texts)
    sub_sentences = vectorizer.get_feature_names_out()
    
    print(f"Extracted {len(sub_sentences)} unique sub-sentences")
    return list(sub_sentences)

def compute_embeddings(sentences, model, batch_size=256):
    """Compute embeddings for sentences using SentenceTransformer"""
    print(f"\nComputing embeddings for {len(sentences)} sentences...")
    
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding"):
        batch = sentences[i:i+batch_size]
        batch_embeddings = model.encode(
            batch, 
            normalize_embeddings=True,
            convert_to_tensor=False,
            show_progress_bar=False,
            batch_size=batch_size
        )
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    print(f"Embedding shape: {embeddings.shape}")
    return embeddings

def cluster_and_save(embeddings, sentences, num_prototypes, output_dir, dataset_name):
    """
    Perform K-means clustering and save results.
    """
    print(f"\nClustering into {num_prototypes} prototypes...")
    print(f"Embedding shape: {embeddings.shape}")
    
    # K-means clustering with mini-batch for efficiency
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=num_prototypes,
        random_state=42,
        batch_size=1000,
        max_iter=100,
        verbose=1,
        n_init=3
    )
    
    print("Fitting K-means...")
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    print(f"Clustering complete.")
    
    
    # Create output directory structure
    output_path = output_dir / "all-mpnet-base-v2"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save cluster centers
    centers_file = output_path / f"{dataset_name}_cluster_{num_prototypes}_centers.npy"
    np.save(centers_file, centers)
    print(f"✓ Saved cluster centers to: {centers_file}")
    
    # Find representative sentences for each cluster
    print("\nFinding representative sentences for each prototype...")
    representative_sentences = {}
    
    for proto_id in range(num_prototypes):
        # Get all sentences in this cluster
        cluster_mask = labels == proto_id
        cluster_sentences = [sentences[i] for i in np.where(cluster_mask)[0]]
        cluster_embeddings = embeddings[cluster_mask]
        
        # Find top-k closest sentences to cluster center
        if len(cluster_sentences) > 0:
            # Compute distances to center
            distances = np.linalg.norm(cluster_embeddings - centers[proto_id], axis=1)
            # Get top 40 closest (as in paper)
            top_k = min(40, len(cluster_sentences))
            top_indices = np.argsort(distances)[:top_k]
            
            representative_sentences[proto_id] = [cluster_sentences[i] for i in top_indices]
        else:
            representative_sentences[proto_id] = []
        
        print(f"Prototype {proto_id}: {len(representative_sentences[proto_id])} representative sentences")
    
    # Save representative sentences as CSV
    # Format: rows = prototypes, columns = top-k sentences
    max_representatives = max(len(v) for v in representative_sentences.values())
    sentence_matrix = []
    
    for proto_id in range(num_prototypes):
        row = representative_sentences[proto_id]
        # Pad with empty strings if needed
        row += [''] * (max_representatives - len(row))
        sentence_matrix.append(row)
    
    df = pd.DataFrame(sentence_matrix)
    sentences_file = output_path / f"{dataset_name}_cluster_{num_prototypes}_to_sub_sentence.csv"
    df.to_csv(sentences_file, header=False)
    print(f"✓ Saved representative sentences to: {sentences_file}")
    
    # Print sample prototypes
    print("\n" + "="*80)
    print("Sample Prototypes (first 3):")
    print("="*80)
    for proto_id in range(min(3, num_prototypes)):
        print(f"\nPrototype {proto_id}:")
        for i, sent in enumerate(representative_sentences[proto_id][:5]):
            print(f"  {i+1}. {sent}")
    
    return centers, representative_sentences

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate prototype initialization for ProtoLens')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['Yelp', 'Amazon', 'IMDB', 'Hotel', 'Steam'],
                        help='Dataset name')
    parser.add_argument('--num_prototypes', type=int, default=20,
                        help='Number of prototypes (default: 20)')
    parser.add_argument('--window_size', type=int, default=5,
                        help='Window size for n-grams (default: 5)')
    parser.add_argument('--max_samples', type=int, default=50000,
                        help='Max samples to use for sub-sentence extraction (default: 50000)')
    parser.add_argument('--max_sub_sentences', type=int, default=30000,
                        help='Max sub-sentences to cluster (default: 30000)')
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent / "Datasets"
    dataset_dir = base_dir / args.dataset
    train_file = dataset_dir / "train.csv"
    
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        return
    
    print("="*80)
    print(f"Generating Prototypes for {args.dataset}")
    print("="*80)
    print(f"Number of prototypes: {args.num_prototypes}")
    print(f"Window size: {args.window_size}")
    print(f"Max samples for extraction: {args.max_samples}")
    print(f"Max sub-sentences to cluster: {args.max_sub_sentences}")
    
    # Load training data
    print("\n" + "="*80)
    print("STEP 1: Loading training data")
    print("="*80)
    df = pd.read_csv(train_file)
    print(f"Loaded {len(df)} training samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Get text column
    text_col = 'review' if 'review' in df.columns else 'text'
    texts = df[text_col].tolist()
    
    # Extract sub-sentences
    print("\n" + "="*80)
    print("STEP 2: Extracting sub-sentences")
    print("="*80)
    sub_sentences = extract_sub_sentences(texts, window_size=args.window_size, 
                                          max_samples=args.max_samples)
    
    # Limit number of sub-sentences for efficiency
    if len(sub_sentences) > args.max_sub_sentences:
        print(f"Sampling {args.max_sub_sentences} sub-sentences from {len(sub_sentences)}...")
        np.random.seed(42)
        indices = np.random.choice(len(sub_sentences), args.max_sub_sentences, replace=False)
        sub_sentences = [sub_sentences[i] for i in indices]
    
    # Load SentenceTransformer model
    print("\n" + "="*80)
    print("STEP 3: Loading SentenceTransformer model")
    print("="*80)
    # Use CPU to avoid CUDA compatibility issues
    device = 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
    
    # Compute embeddings
    print("\n" + "="*80)
    print("STEP 4: Computing embeddings")
    print("="*80)
    embeddings = compute_embeddings(sub_sentences, model, batch_size=256)
    
    # Cluster and save
    print("\n" + "="*80)
    print("STEP 5: Clustering and saving")
    print("="*80)
    centers, representatives = cluster_and_save(
        embeddings, sub_sentences, args.num_prototypes, 
        dataset_dir, args.dataset
    )
    
    print("\n" + "="*80)
    print("✓ Prototype generation complete!")
    print("="*80)
    print(f"\nGenerated files:")
    output_path = dataset_dir / "all-mpnet-base-v2"
    print(f"  - {output_path / f'{args.dataset}_cluster_{args.num_prototypes}_centers.npy'}")
    print(f"  - {output_path / f'{args.dataset}_cluster_{args.num_prototypes}_to_sub_sentence.csv'}")

if __name__ == "__main__":
    main()
