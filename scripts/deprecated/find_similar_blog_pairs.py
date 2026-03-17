#!/usr/bin/env python3
"""
Find similar blog pairs.
Ensures all pairs are distinct - no blog appears in multiple pairs.
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cs4.config import Config
from cs4.utils.embedding_utils import (
    load_or_create_embeddings,
    find_similar_pairs_distinct,
    save_pairs_to_csv
)

def main():
    parser = argparse.ArgumentParser(
        description="Find similar blog pairs"
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to input CSV with blogs"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV with blog pairs"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=10000,
        help="Maximum number of blogs to process (default: 10000)"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=25,
        help="Maximum number of pairs to find (default: 25)"
    )
    parser.add_argument(
        "--similarity-lower",
        type=float,
        default=0.70,
        help="Lower bound for similar pairs (default: 0.70)"
    )
    parser.add_argument(
        "--similarity-upper",
        type=float,
        default=0.85,
        help="Upper bound for similar pairs (default: 0.85)"
    )
    parser.add_argument(
        "--model",
        default="all-mpnet-base-v2",
        help="Sentence transformer model (default: all-mpnet-base-v2)"
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=f"Directory to cache embeddings (default: {Config.OUTPUTS_DIR} from config)"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=1000,
        help="Minimum word count per blog text (default: 1000)"
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=1500,
        help="Maximum word count per blog text (default: 1500)"
    )
    
    args = parser.parse_args()
    
    print(f"Finding similar blog pairs (distinct pairs only)")
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Max size: {args.max_size}")
    print(f"Similarity range: {args.similarity_lower}-{args.similarity_upper}")
    print(f"Word count filter: {args.min_words}-{args.max_words} words")
    
    # Validate input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file {args.input_path} does not exist")
        sys.exit(1)
    
    try:
        # Load or create embeddings
        sentences, embeddings = load_or_create_embeddings(
            file_path=args.input_path,
            max_size=args.max_size,
            model_name=args.model,
            cache_dir=args.cache_dir,
            min_words=args.min_words,
            max_words=args.max_words
        )
        
        # Find similar pairs (ensuring distinct pairs)
        pairs = find_similar_pairs_distinct(
            sentences=sentences,
            embeddings=embeddings,
            max_pairs=args.max_pairs,
            similarity_lower=args.similarity_lower,
            similarity_upper=args.similarity_upper
        )
        
        # Save results
        save_pairs_to_csv(pairs, args.output_path)
        
        print("Similar pair finding complete!")
        
    except Exception as e:
        print(f"Failed to find pairs: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
