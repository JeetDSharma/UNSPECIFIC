"""
Embedding utilities for blog similarity and pairing.
Based on working story_NN_search.py logic.
"""

import csv
import pickle
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from cs4.config import Config


def read_texts(file_path, max_size, min_words=None, max_words=None):
    """Read texts from CSV file with optional word count filtering
    
    Args:
        file_path: Path to CSV file
        max_size: Maximum number of texts to read
        min_words: Minimum word count per text (inclusive, optional)
        max_words: Maximum word count per text (inclusive, optional)
    """
    # Increase CSV field size limit to handle large text fields
    csv.field_size_limit(sys.maxsize)
    
    texts = []
    skipped = 0
    with open(file_path, encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            txt = row.get("text", "").strip()
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


def load_or_create_embeddings(
    file_path, 
    max_size=1000, 
    model_name='all-mpnet-base-v2',
    cache_dir=None,
    min_words=None,
    max_words=None
):
    """
    Load embeddings from cache or create new ones.
    
    Args:
        file_path: Path to CSV file with blogs
        max_size: Maximum number of blogs to process
        model_name: Sentence transformer model name
        cache_dir: Directory to cache embeddings
        min_words: Minimum word count per text (optional)
        max_words: Maximum word count per text (optional)
    """
    # Use Config.OUTPUTS_DIR if no cache_dir specified
    if cache_dir is None:
        cache_dir = Config.OUTPUTS_DIR
    
    model = SentenceTransformer(model_name)
    
    # Build cache filename with word count filters if specified
    cache_filename = f'embeddings-{model_name.replace("/", "_")}-size-{max_size}'
    if min_words is not None:
        cache_filename += f'-min{min_words}'
    if max_words is not None:
        cache_filename += f'-max{max_words}'
    cache_filename += '.pkl'
    
    embedding_cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        print("Encode the corpus. This might take a while")
        if min_words is not None or max_words is not None:
            print(f"Filtering texts: min_words={min_words}, max_words={max_words}")
        corpus_sentences = read_texts(file_path, max_size, min_words, max_words)
        corpus_embeddings = model.encode(
            corpus_sentences, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )

        print("Store file on disc")
        os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)

        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({
                'sentences': corpus_sentences, 
                'embeddings': corpus_embeddings
            }, fOut)
    else:
        print("Load pre-computed embeddings from disc")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            corpus_sentences = cache_data['sentences']
            corpus_embeddings = cache_data['embeddings']

    print(f"Corpus loaded with {len(corpus_sentences)} sentences / embeddings")
    return corpus_sentences, corpus_embeddings


def find_dissimilar_pairs(
    sentences, 
    embeddings, 
    max_pairs=25, 
    dissimilarity_lower=0.40, 
    dissimilarity_upper=0.50
):
    """Find dissimilar pairs using normalized embeddings"""
    print("Finding dissimilar pairs...")
    
    # Normalize embeddings
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    pairs = []

    for i in tqdm(range(len(sentences))):
        # Sample candidates
        if i + 1 >= len(sentences):
            break
            
        candidates = np.random.choice(
            range(i + 1, len(sentences)),
            size=min(50, len(sentences) - i - 1),
            replace=False
        )
        
        for idx2 in candidates:
            # Compute similarity
            similarity = float(np.dot(normalized_embeddings[i], normalized_embeddings[idx2]))
            
            # Check if similarity is within the dissimilar range
            if dissimilarity_lower <= similarity <= dissimilarity_upper:
                pairs.append({
                    'blog_1_id': i,
                    'blog_2_id': idx2,
                    'blog_1_text': sentences[i],
                    'blog_2_text': sentences[idx2],
                    'similarity': similarity
                })
                
                if len(pairs) >= max_pairs:
                    break
        
        if len(pairs) >= max_pairs:
            break

    print(f"Found {len(pairs)} dissimilar pairs")
    return pairs


def find_dissimilar_pairs_distinct(
    sentences, 
    embeddings, 
    max_pairs=25, 
    dissimilarity_lower=None, 
    dissimilarity_upper=0.50
):
    """Find dissimilar pairs ensuring all pairs are distinct (no blog appears in multiple pairs)"""
    # Use Config defaults if not specified
    if dissimilarity_lower is None:
        dissimilarity_lower = Config.DISSIMILAR_THRESHOLD
    
    print("Finding dissimilar pairs (distinct pairs only)...")
    
    # Normalize embeddings
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    pairs = []
    used_ids = set()  # Track which blog IDs have been used in pairs

    for i in tqdm(range(len(sentences))):
        if len(pairs) >= max_pairs:
            break
        if i in used_ids:
            continue

        # Sample candidates from unused IDs only
        available_candidates = [j for j in range(i + 1, len(sentences)) if j not in used_ids]
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
            
            # Check if similarity is within the dissimilar range
            if dissimilarity_lower <= similarity <= dissimilarity_upper:
                pairs.append({
                    'blog_1_id': i,
                    'blog_2_id': idx2,
                    'blog_1_text': sentences[i],
                    'blog_2_text': sentences[idx2],
                    'similarity': round(similarity, 3)  # Round to 3 decimals like in notebook
                })
                
                # Mark both IDs as used to ensure distinct pairs
                used_ids.add(i)
                used_ids.add(idx2)
                break
        
        if len(pairs) >= max_pairs:
            break

    print(f"Found {len(pairs)} dissimilar pairs (all distinct)")
    return pairs


def find_similar_pairs_distinct(
    sentences, 
    embeddings, 
    max_pairs=25, 
    similarity_lower=0.70, 
    similarity_upper=0.85
):
    """Find similar pairs ensuring all pairs are distinct (no blog appears in multiple pairs)"""
    print("Finding similar pairs (distinct pairs only)...")
    
    # Normalize embeddings
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, None]

    pairs = []
    used_ids = set()  # Track which blog IDs have been used in pairs

    for i in tqdm(range(len(sentences))):
        if len(pairs) >= max_pairs:
            break
        if i in used_ids:
            continue

        # Sample candidates from unused IDs only
        available_candidates = [j for j in range(i + 1, len(sentences)) if j not in used_ids]
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
            if similarity_lower <= similarity <= similarity_upper:
                pairs.append({
                    'blog_1_id': i,
                    'blog_2_id': idx2,
                    'blog_1_text': sentences[i],
                    'blog_2_text': sentences[idx2],
                    'similarity': round(similarity, 3)  # Round to 3 decimals like in notebook
                })
                
                # Mark both IDs as used to ensure distinct pairs
                used_ids.add(i)
                used_ids.add(idx2)
                break
        
        if len(pairs) >= max_pairs:
            break

    print(f"Found {len(pairs)} similar pairs (all distinct)")
    return pairs


def save_pairs_to_csv(pairs, output_path):
    """Save pairs to CSV file without pandas"""
    import csv
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not pairs:
        print("No pairs to save")
        return
    
    # Write CSV manually
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['blog_1_id', 'blog_2_id', 'blog_1_text', 'blog_2_text', 'similarity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair)
    
    print(f"Saved {len(pairs)} pairs to {output_path}")
    
    # Print statistics
    if pairs:
        similarities = [pair['similarity'] for pair in pairs]
        print(f"Similarity range: [{min(similarities):.3f}, {max(similarities):.3f}]")
        print(f"Mean similarity: {sum(similarities)/len(similarities):.3f}")
