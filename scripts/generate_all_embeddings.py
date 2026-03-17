#!/usr/bin/env python3
"""
Generate embeddings for ALL blogs in the CSV without any filtering.
This creates a master embeddings file that can be used for testing different configurations.
"""

import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

def count_words(text):
    """Count words in text"""
    if pd.isna(text):
        return 0
    return len(str(text).split())

def generate_all_embeddings(csv_path, model_name='all-mpnet-base-v2', cache_dir='outputs'):
    """Generate embeddings for all blogs in the CSV"""
    
    print(f"Loading blogs from: {csv_path}")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} blogs from CSV")
    
    # Filter out any blogs with no text
    df = df.dropna(subset=['text'])
    print(f"After removing empty texts: {len(df)} blogs")
    
    # Get all texts
    all_texts = df['text'].tolist()
    
    # Count words for each text (for later filtering)
    word_counts = [count_words(text) for text in all_texts]
    
    print(f"Word count statistics:")
    print(f"  Min: {min(word_counts)} words")
    print(f"  Max: {max(word_counts)} words")
    print(f"  Mean: {np.mean(word_counts):.1f} words")
    print(f"  Median: {np.median(word_counts):.1f} words")
    
    # Initialize the model
    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    print(f"Generating embeddings for {len(all_texts)} blogs...")
    print("This might take a while...")
    
    embeddings = model.encode(
        all_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )
    
    print(f"Embeddings generated. Shape: {embeddings.shape}")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save embeddings with metadata
    output_file = os.path.join(cache_dir, f'embeddings-{model_name}-ALL-{len(df)}blogs.pkl')
    
    data_to_save = {
        'sentences': all_texts,
        'embeddings': embeddings,
        'word_counts': word_counts,
        'blog_ids': df.index.tolist(),  # Save original indices
        'model_name': model_name,
        'total_blogs': len(df),
        'source_csv': csv_path
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    print(f"\nAll embeddings saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    return output_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate embeddings for all blogs')
    parser.add_argument('--csv_path', required=True, help='Path to blog CSV file')
    parser.add_argument('--model', default='all-mpnet-base-v2', help='Model name')
    parser.add_argument('--cache_dir', default='outputs', help='Cache directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return
    
    output_file = generate_all_embeddings(
        csv_path=args.csv_path,
        model_name=args.model,
        cache_dir=args.cache_dir
    )
    
    print(f"\n✅ Master embeddings file created: {output_file}")
    print("You can now use this file to test different configurations quickly!")

if __name__ == "__main__":
    main()
