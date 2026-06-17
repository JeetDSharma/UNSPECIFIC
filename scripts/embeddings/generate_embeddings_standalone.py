#!/usr/bin/env python3
"""
generate_embeddings_standalone.py
---------------
Standalone script to generate embeddings for blog posts.
"""

import argparse
import csv
import pickle
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def read_texts(file_path, max_size, min_words=None, max_words=None, text_column="text"):
    """Read texts from CSV file with optional word count filtering"""
    csv.field_size_limit(sys.maxsize)
    
    texts = []
    skipped = 0
    with open(file_path, encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = row.get(text_column, "").strip()
            if txt:
                # Apply word count filtering if specified
                if min_words is not None or max_words is not None:
                    word_count = len(txt.split())
                    if min_words is not None and word_count < min_words:
                        skipped += 1
                        continue
                    if max_words is not None and word_count > max_words:
                        skipped += 1
                        continue
                
                texts.append(txt)
            if len(texts) >= max_size:
                break
    
    if skipped > 0:
        print(f"Filtered out {skipped} texts based on word count constraints")
    
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog_csv", required=True, help="Path to blog CSV file")
    ap.add_argument(
        "--text_column",
        default="text",
        help="CSV column with article body (default: text; use e.g. content for news)",
    )
    ap.add_argument("--max_size", type=int, default=10000, help="Maximum blogs to process")
    ap.add_argument("--min_words", type=int, help="Minimum word count filter")
    ap.add_argument("--max_words", type=int, help="Maximum word count filter")
    ap.add_argument("--model", default="all-mpnet-base-v2", help="Sentence transformer model")
    ap.add_argument("--output_dir", default="outputs", help="Output directory for embeddings")
    args = ap.parse_args()

    # Validate input file exists
    if not os.path.exists(args.blog_csv):
        print(f"Error: Blog CSV file not found: {args.blog_csv}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating embeddings for blogs from: {args.blog_csv}")
    print(f"Model: {args.model}")
    print(f"Max size: {args.max_size}")
    
    if args.min_words or args.max_words:
        print(f"Word count filter: min={args.min_words}, max={args.max_words}")

    # Build cache filename (include text column so different columns do not share cache)
    col_tag = args.text_column.replace("/", "_")
    cache_filename = f'embeddings-{args.model.replace("/", "_")}-col-{col_tag}-size-{args.max_size}'
    if args.min_words is not None:
        cache_filename += f'-min{args.min_words}'
    if args.max_words is not None:
        cache_filename += f'-max{args.max_words}'
    cache_filename += '.pkl'
    
    embedding_cache_path = os.path.join(args.output_dir, cache_filename)

    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        print("Encoding the corpus. This might take a while...")
        
        # Load model
        model = SentenceTransformer(args.model)
        
        # Read texts
        corpus_sentences = read_texts(
            args.blog_csv, args.max_size, args.min_words, args.max_words, args.text_column
        )
        
        # Generate embeddings
        corpus_embeddings = model.encode(
            corpus_sentences, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )

        print("Storing embeddings to disk...")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({
                'sentences': corpus_sentences, 
                'embeddings': corpus_embeddings
            }, fOut)
    else:
        print("Loading pre-computed embeddings from disk...")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            corpus_sentences = cache_data['sentences']
            corpus_embeddings = cache_data['embeddings']

    print(f"Corpus loaded with {len(corpus_sentences)} sentences / embeddings")
    print(f"Embedding shape: {corpus_embeddings.shape}")
    print(f"Embeddings saved to: {embedding_cache_path}")


if __name__ == "__main__":
    main()
