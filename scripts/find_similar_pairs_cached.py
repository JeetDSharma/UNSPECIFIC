#!/usr/bin/env python3
"""
Find similar blog pairs using cached embeddings.
Applies word count filtering during pair finding instead of during embedding generation.
"""

import argparse
import csv
import pickle
import os
import sys
import numpy as np
from tqdm import tqdm


def count_words(text):
    """Count words in text"""
    return len(text.split())


def find_pairs_with_criteria(
    sentences, 
    embeddings, 
    min_similarity=0.7, 
    max_similarity=0.85,
    min_words=800,
    max_words=2000,
    max_pairs=25
):
    """Find pairs meeting similarity and word count criteria"""
    print(f"Finding pairs with similarity {min_similarity}-{max_similarity} and word count {min_words}-{max_words}...")
    
    # Pre-filter sentences by word count
    valid_indices = []
    for i, text in enumerate(sentences):
        word_count = count_words(text)
        if min_words <= word_count <= max_words:
            valid_indices.append(i)
    
    print(f"Found {len(valid_indices)} texts with word count {min_words}-{max_words}")
    
    if len(valid_indices) < 2:
        print("Not enough texts meet the word count criteria")
        return []

    # Normalize embeddings
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    pairs = []
    used_ids = set()  # Track which blog IDs have been used in pairs

    for i in tqdm(valid_indices):
        if len(pairs) >= max_pairs:
            break
        if i in used_ids:
            continue

        # Sample candidates from unused valid IDs only
        available_candidates = [j for j in valid_indices if j not in used_ids and j != i]
        if not available_candidates:
            continue
            
        candidates = np.random.choice(
            available_candidates,
            size=min(50, len(available_candidates)),
            replace=False
        )
        
        for idx2 in candidates:
            # Compute similarity
            similarity = float(np.dot(normalized_embeddings[i], normalized_embeddings[idx2]))
            
            # Check if similarity is within the similar range
            if min_similarity <= similarity <= max_similarity:
                pairs.append({
                    'blog_1_id': i,
                    'blog_2_id': idx2,
                    'blog_1_text': sentences[i],
                    'blog_2_text': sentences[idx2],
                    'similarity': round(similarity, 3),
                    'blog_1_word_count': count_words(sentences[i]),
                    'blog_2_word_count': count_words(sentences[idx2])
                })
                
                # Mark both IDs as used to ensure distinct pairs
                used_ids.add(i)
                used_ids.add(idx2)
                break
        
        if len(pairs) >= max_pairs:
            break

    print(f"Found {len(pairs)} pairs meeting all criteria")
    return pairs


def save_pairs_to_csv(pairs, output_path):
    """Save pairs to CSV file"""
    if not pairs:
        print("No pairs to save")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['blog_1_id', 'blog_2_id', 'blog_1_text', 'blog_2_text', 'similarity', 
                     'blog_1_word_count', 'blog_2_word_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)
    
    print(f"Saved {len(pairs)} pairs to {output_path}")
    
    # Print statistics
    if pairs:
        similarities = [pair['similarity'] for pair in pairs]
        word_counts_1 = [pair['blog_1_word_count'] for pair in pairs]
        word_counts_2 = [pair['blog_2_word_count'] for pair in pairs]
        
        print(f"\nStatistics:")
        print(f"Similarity range: [{min(similarities):.3f}, {max(similarities):.3f}]")
        print(f"Mean similarity: {sum(similarities)/len(similarities):.3f}")
        print(f"Word count range (blog 1): [{min(word_counts_1)}, {max(word_counts_1)}]")
        print(f"Word count range (blog 2): [{min(word_counts_2)}, {max(word_counts_2)}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="Path to cached embeddings pickle file")
    ap.add_argument("--output", default="outputs/similar_pairs.csv", help="Output CSV file")
    ap.add_argument("--min_similarity", type=float, default=0.7, help="Minimum similarity")
    ap.add_argument("--max_similarity", type=float, default=0.85, help="Maximum similarity")
    ap.add_argument("--min_words", type=int, default=800, help="Minimum word count")
    ap.add_argument("--max_words", type=int, default=2000, help="Maximum word count")
    ap.add_argument("--max_pairs", type=int, default=25, help="Maximum number of pairs to find")
    args = ap.parse_args()

    # Validate input file exists
    if not os.path.exists(args.embeddings):
        print(f"Error: Embeddings file not found: {args.embeddings}")
        return

    print(f"Loading cached embeddings from: {args.embeddings}")
    
    # Load embeddings
    with open(args.embeddings, "rb") as f:
        data = pickle.load(f)
    
    sentences = data["sentences"]
    embeddings = data["embeddings"]
    
    print(f"Loaded {len(sentences)} sentences and embeddings")

    # Find pairs
    pairs = find_pairs_with_criteria(
        sentences=sentences,
        embeddings=embeddings,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
        min_words=args.min_words,
        max_words=args.max_words,
        max_pairs=args.max_pairs
    )

    # Save results
    if pairs:
        save_pairs_to_csv(pairs, args.output)
    else:
        print("No pairs found matching the criteria")


if __name__ == "__main__":
    main()
