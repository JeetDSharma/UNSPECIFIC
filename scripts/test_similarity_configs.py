#!/usr/bin/env python3
"""
Test different similarity and word count configurations using master embeddings.
This allows quick experimentation without re-generating embeddings.
"""

import pickle
import os
import csv
import numpy as np
from tqdm import tqdm
import argparse

def count_words(text):
    """Count words in text"""
    return len(text.split())

def find_pairs_with_config(
    sentences, 
    embeddings, 
    word_counts,
    min_similarity=0.7, 
    max_similarity=0.85,
    min_words=800,
    max_words=2000,
    max_pairs=25,
    sample_size=500
):
    """Find pairs meeting specific criteria"""
    
    print(f"Finding pairs with similarity {min_similarity}-{max_similarity} and word count {min_words}-{max_words}")
    
    # Filter indices by word count
    valid_indices = []
    for i, word_count in enumerate(word_counts):
        if min_words <= word_count <= max_words:
            valid_indices.append(i)
    
    print(f"Found {len(valid_indices)} texts with word count {min_words}-{max_words}")
    
    if len(valid_indices) < 2:
        print("Not enough texts meet the word count criteria")
        return []

    # Normalize embeddings
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    pairs = []
    used_ids = set()

    for i in tqdm(valid_indices, desc="Searching pairs"):
        if len(pairs) >= max_pairs:
            break
        if i in used_ids:
            continue

        # Sample candidates from unused valid IDs
        available_candidates = [j for j in valid_indices if j not in used_ids and j != i]
        if not available_candidates:
            continue
            
        # Limit candidates for efficiency (or use 'all' for exhaustive search)
        if sample_size == -1:  # -1 means exhaustive search
            candidates = available_candidates
        else:
            candidates = np.random.choice(
                available_candidates,
                size=min(sample_size, len(available_candidates)),
                replace=False
            )
        
        for idx2 in candidates:
            # Compute similarity
            similarity = float(np.dot(normalized_embeddings[i], normalized_embeddings[idx2]))
            
            # Check if similarity is within the target range
            if min_similarity <= similarity <= max_similarity:
                pairs.append({
                    'blog_1_id': i,
                    'blog_2_id': idx2,
                    'blog_1_text': sentences[i],
                    'blog_2_text': sentences[idx2],
                    'similarity': round(similarity, 3),
                    'blog_1_word_count': word_counts[i],
                    'blog_2_word_count': word_counts[idx2]
                })
                
                # Mark both IDs as used
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
        
        print(f"\n📊 Statistics:")
        print(f"  Similarity range: [{min(similarities):.3f}, {max(similarities):.3f}]")
        print(f"  Mean similarity: {sum(similarities)/len(similarities):.3f}")
        print(f"  Word count range (blog 1): [{min(word_counts_1)}, {max(word_counts_1)}]")
        print(f"  Word count range (blog 2): [{min(word_counts_2)}, {max(word_counts_2)}]")

def test_config(embeddings_file, output_root_dir, config_name, min_sim, max_sim, min_words, max_words, max_pairs=25, sample_size=500):
    """Test a specific configuration"""
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing Configuration: {config_name}")
    print(f"   Similarity: {min_sim}-{max_sim}")
    print(f"   Word count: {min_words}-{max_words}")
    print(f"   Target pairs: {max_pairs}")
    print(f"{'='*60}")
    
    # Load embeddings
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    sentences = data['sentences']
    embeddings = data['embeddings']
    word_counts = data['word_counts']
    
    print(f"Loaded {len(sentences)} blogs from master embeddings")
    
    # Find pairs
    pairs = find_pairs_with_config(
        sentences=sentences,
        embeddings=embeddings,
        word_counts=word_counts,
        min_similarity=min_sim,
        max_similarity=max_sim,
        min_words=min_words,
        max_words=max_words,
        max_pairs=max_pairs,
        sample_size=sample_size
    )
    
    # Save results
    if pairs:
        config_dir_name = config_name.replace(' ', '_').lower()
        config_output_dir = os.path.join(output_root_dir, config_dir_name)
        os.makedirs(config_output_dir, exist_ok=True)

        output_file = os.path.join(config_output_dir, "pairs.csv")
        save_pairs_to_csv(pairs, output_file)
        print(f"✅ Configuration '{config_name}' completed successfully!")
    else:
        print(f"❌ Configuration '{config_name}' found no matching pairs")
    
    return len(pairs)

def main():
    parser = argparse.ArgumentParser(description='Test different similarity configurations')
    parser.add_argument('--embeddings', required=True, help='Path to master embeddings file')
    parser.add_argument('--config', default=None, help='(Optional) Test a preset config name or "all"')
    parser.add_argument(
        '--output-root-dir',
        default='data/outputs/common_constraints/blog_similarity_pairs_experiments_master_embeddings',
        help='Root directory to write outputs; each config writes into its own subfolder'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=500,
        help='Number of candidates to sample per blog (use -1 for exhaustive search of all candidates)'
    )

    parser.add_argument('--min-similarity', type=float, default=None, help='Minimum cosine similarity (direct mode)')
    parser.add_argument('--max-similarity', type=float, default=None, help='Maximum cosine similarity (direct mode)')
    parser.add_argument('--min-words', type=int, default=None, help='Minimum word count (direct mode)')
    parser.add_argument('--max-words', type=int, default=None, help='Maximum word count (direct mode)')
    parser.add_argument('--max-pairs', type=int, default=25, help='Maximum number of pairs to output')
    parser.add_argument(
        '--run-name',
        default=None,
        help='(Optional) Folder name for direct mode output. If omitted, auto-generated from constraints.'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.embeddings):
        print(f"Error: Embeddings file not found: {args.embeddings}")
        return

    os.makedirs(args.output_root_dir, exist_ok=True)
    
    # Presets are kept intentionally minimal; prefer direct CLI args.
    presets = {
        "Original_0.7_0.85_800_2000": (0.7, 0.85, 800, 2000),
        "Original_0.7_0.85_1000_2000": (0.7, 0.85, 1000, 2000),
    }

    print("🚀 Starting configuration testing...")
    print(f"Using master embeddings: {args.embeddings}")

    results = []

    direct_mode = any(
        v is not None
        for v in (args.min_similarity, args.max_similarity, args.min_words, args.max_words)
    )

    if direct_mode:
        if args.min_similarity is None or args.max_similarity is None or args.min_words is None or args.max_words is None:
            print("Error: direct mode requires --min-similarity, --max-similarity, --min-words, --max-words")
            return

        if args.run_name is not None:
            config_name = args.run_name
        else:
            config_name = f"sim{args.min_similarity}-{args.max_similarity}_wc{args.min_words}-{args.max_words}"

        pairs_found = test_config(
            args.embeddings,
            args.output_root_dir,
            config_name,
            args.min_similarity,
            args.max_similarity,
            args.min_words,
            args.max_words,
            max_pairs=args.max_pairs,
            sample_size=args.sample_size,
        )
        results.append((config_name, pairs_found))

    else:
        # Preset mode (optional)
        if args.config is None:
            print("Error: provide either direct args (--min-similarity/--max-similarity/--min-words/--max-words) or --config")
            print("Available presets:")
            for name in presets.keys():
                print(f"  - {name}")
            return

        if args.config.lower() == 'all':
            for config_name, (min_sim, max_sim, min_words, max_words) in presets.items():
                pairs_found = test_config(
                    args.embeddings,
                    args.output_root_dir,
                    config_name,
                    min_sim,
                    max_sim,
                    min_words,
                    max_words,
                    max_pairs=args.max_pairs,
                    sample_size=args.sample_size,
                )
                results.append((config_name, pairs_found))
        else:
            matched = None
            for name in presets.keys():
                if args.config.lower() in name.lower():
                    matched = name
                    break

            if matched is None:
                print(f"Error: preset config '{args.config}' not found")
                print("Available presets:")
                for name in presets.keys():
                    print(f"  - {name}")
                return

            min_sim, max_sim, min_words, max_words = presets[matched]
            pairs_found = test_config(
                args.embeddings,
                args.output_root_dir,
                matched,
                min_sim,
                max_sim,
                min_words,
                max_words,
                max_pairs=args.max_pairs,
                sample_size=args.sample_size,
            )
            results.append((matched, pairs_found))
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY OF RESULTS")
    print(f"{'='*60}")
    print(f"{'Configuration':<30} {'Pairs Found':<15}")
    print("-" * 45)
    for config_name, pairs_found in results:
        print(f"{config_name:<30} {pairs_found:<15}")
    
    # Find best configuration
    if results:
        best_config = max(results, key=lambda x: x[1])
        print(f"\n🏆 Best configuration: {best_config[0]} with {best_config[1]} pairs")
        
        if best_config[1] >= 25:
            print("✅ Target of 25 pairs achieved!")
        else:
            print(f"⚠️  Still need {25 - best_config[1]} more pairs to reach target")

if __name__ == "__main__":
    main()
