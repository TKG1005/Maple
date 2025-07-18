#!/usr/bin/env python3
"""
Demo script for move embedding functionality.
Shows how to use the move embedding system for Pokemon move analysis.
"""

from __future__ import annotations

import numpy as np
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.move_embedding import MoveEmbeddingGenerator


def analyze_move_similarities(move_embeddings: dict[str, np.ndarray], 
                            query_move: str, 
                            top_k: int = 5) -> list[tuple[str, float]]:
    """
    Find moves most similar to a query move based on embedding similarity.
    
    Args:
        move_embeddings: Dictionary of move embeddings
        query_move: Name of the move to find similarities for
        top_k: Number of top similar moves to return
        
    Returns:
        List of tuples (move_name, similarity_score)
    """
    if query_move not in move_embeddings:
        print(f"Move '{query_move}' not found in embeddings")
        return []
    
    query_embedding = move_embeddings[query_move]
    similarities = []
    
    for move_name, embedding in move_embeddings.items():
        if move_name == query_move or move_name.startswith('move_'):
            continue  # Skip self and move_id entries
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        similarities.append((move_name, similarity))
    
    # Sort by similarity (descending) and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def analyze_move_types(move_embeddings: dict[str, np.ndarray], 
                      feature_names: list[str]) -> dict[str, list[str]]:
    """
    Analyze moves by their types based on embedding features.
    
    Args:
        move_embeddings: Dictionary of move embeddings
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping type names to lists of move names
    """
    type_features = [name for name in feature_names if name.startswith('type_')]
    type_moves = {type_name.replace('type_', ''): [] for type_name in type_features}
    
    for move_name, embedding in move_embeddings.items():
        if move_name.startswith('move_'):
            continue  # Skip move_id entries
        
        # Find the type with the highest activation
        type_activations = []
        for i, feature_name in enumerate(feature_names):
            if feature_name.startswith('type_'):
                type_name = feature_name.replace('type_', '')
                type_activations.append((type_name, embedding[i]))
        
        if type_activations:
            primary_type = max(type_activations, key=lambda x: x[1])[0]
            type_moves[primary_type].append(move_name)
    
    return type_moves


def main():
    parser = argparse.ArgumentParser(description="Demo enhanced move embedding functionality")
    parser.add_argument("--moves-csv", default="config/moves.csv", 
                       help="Path to moves CSV file")
    parser.add_argument("--query-move", default="はたく", 
                       help="Move to find similarities for")
    parser.add_argument("--top-k", type=int, default=5, 
                       help="Number of similar moves to show")
    parser.add_argument("--analyze-types", action="store_true", 
                       help="Analyze moves by type")
    parser.add_argument("--fusion-strategy", default="balanced", 
                       choices=["concatenate", "balanced", "weighted"],
                       help="Fusion strategy for combining features")
    parser.add_argument("--japanese-model", action="store_true", default=True,
                       help="Use Japanese-specific model")
    parser.add_argument("--semantic-search", action="store_true",
                       help="Perform semantic search demo")
    
    args = parser.parse_args()
    
    print("=== Enhanced Pokemon Move Embedding Demo ===")
    print(f"Loading moves from: {args.moves_csv}")
    print(f"Fusion strategy: {args.fusion_strategy}")
    print(f"Japanese model: {args.japanese_model}")
    
    # Generate embeddings with enhanced options
    generator = MoveEmbeddingGenerator(args.moves_csv, japanese_model=args.japanese_model)
    move_embeddings, feature_names = generator.generate_embeddings(
        fusion_strategy=args.fusion_strategy
    )
    
    print(f"Generated embeddings for {len(move_embeddings)} moves")
    print(f"Feature dimension: {len(feature_names)}")
    
    # Analyze move similarities
    print(f"\n=== Finding moves similar to '{args.query_move}' ===")
    similar_moves = analyze_move_similarities(move_embeddings, args.query_move, args.top_k)
    
    if similar_moves:
        print(f"Top {args.top_k} moves similar to '{args.query_move}':")
        for i, (move_name, similarity) in enumerate(similar_moves, 1):
            print(f"  {i}. {move_name} (similarity: {similarity:.3f})")
    else:
        print(f"No similar moves found for '{args.query_move}'")
    
    # Semantic search demo
    if args.semantic_search:
        print(f"\n=== Semantic Search Demo ===")
        semantic_queries = [
            "攻撃力を上げる技",
            "リフレクター系の技", 
            "回復技",
            "状態異常を与える技",
            "先制技"
        ]
        
        for query in semantic_queries:
            print(f"\nQuery: '{query}'")
            results = generator.semantic_search(query, move_embeddings, top_k=3)
            if results:
                for i, (move_name, similarity) in enumerate(results, 1):
                    print(f"  {i}. {move_name} (similarity: {similarity:.3f})")
            else:
                print("  No results found")
    
    # Analyze by types if requested
    if args.analyze_types:
        print(f"\n=== Move Type Analysis ===")
        type_moves = analyze_move_types(move_embeddings, feature_names)
        
        for type_name, moves in type_moves.items():
            if moves:  # Only show types with moves
                print(f"{type_name.title()}: {len(moves)} moves")
                # Show first few moves as examples
                example_moves = moves[:3]
                if len(moves) > 3:
                    example_moves.append(f"... and {len(moves) - 3} more")
                print(f"  Examples: {', '.join(example_moves)}")
    
    print(f"\n=== Feature Analysis ===")
    print(f"Feature categories:")
    type_features = [name for name in feature_names if name.startswith('type_')]
    category_features = [name for name in feature_names if name.startswith('category_')]
    scaled_features = [name for name in feature_names if name.endswith('_scaled')]
    desc_features = [name for name in feature_names if name.startswith('desc_emb_')]
    
    print(f"  Types: {len(type_features)} features")
    print(f"  Categories: {len(category_features)} features")
    print(f"  Scaled numerical: {len(scaled_features)} features")
    print(f"  Description embeddings: {len(desc_features)} features")
    
    # Show example embedding
    if args.query_move in move_embeddings:
        print(f"\n=== Example Embedding for '{args.query_move}' ===")
        embedding = move_embeddings[args.query_move]
        print(f"Shape: {embedding.shape}")
        print(f"Min value: {embedding.min():.3f}")
        print(f"Max value: {embedding.max():.3f}")
        print(f"Mean: {embedding.mean():.3f}")
        print(f"Standard deviation: {embedding.std():.3f}")
        print(f"First 10 values: {embedding[:10].round(3)}")


if __name__ == "__main__":
    main()