#!/usr/bin/env python3
"""
generate_embeddings.py
---------------
Generate embeddings for blog posts using sentence transformers.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs4.utils.embedding_utils import load_or_create_embeddings
from cs4.utils.config_loader import load_yaml, stamp, fill_vars
from cs4.utils.log_utils import setup_logging, get_logger
from cs4.utils.io_utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to embedding config")
    ap.add_argument("--blog_csv", required=True, help="Path to blog CSV file")
    ap.add_argument("--max_size", type=int, default=10000, help="Maximum blogs to process")
    ap.add_argument("--min_words", type=int, help="Minimum word count filter")
    ap.add_argument("--max_words", type=int, help="Maximum word count filter")
    ap.add_argument("--model", default="all-mpnet-base-v2", help="Sentence transformer model")
    ap.add_argument("--cache_dir", help="Directory to cache embeddings")
    args = ap.parse_args()

    # Setup logging
    logger = get_logger("EmbeddingGenerator")
    
    # Validate input file exists
    if not os.path.exists(args.blog_csv):
        logger.error(f"Blog CSV file not found: {args.blog_csv}")
        return

    logger.info(f"Generating embeddings for blogs from: {args.blog_csv}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Max size: {args.max_size}")
    
    if args.min_words or args.max_words:
        logger.info(f"Word count filter: min={args.min_words}, max={args.max_words}")

    # Generate embeddings
    try:
        sentences, embeddings = load_or_create_embeddings(
            file_path=args.blog_csv,
            max_size=args.max_size,
            model_name=args.model,
            cache_dir=args.cache_dir,
            min_words=args.min_words,
            max_words=args.max_words
        )
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        logger.info(f"Embedding shape: {embeddings.shape}")
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return


if __name__ == "__main__":
    main()
