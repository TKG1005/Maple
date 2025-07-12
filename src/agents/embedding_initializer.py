"""
Pokemon Species Embedding Initializer

This module provides functionality to initialize Pokemon species embeddings
using base stats (種族値) for the first 6 dimensions, with the remaining
dimensions initialized with small random values.
"""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import os


class EmbeddingInitializer:
    """Initialize Pokemon species embeddings with base stats."""
    
    def __init__(self, pokemon_stats_path: str = "config/pokemon_stats.csv"):
        """
        Initialize the embedding initializer.
        
        Args:
            pokemon_stats_path: Path to Pokemon stats CSV file
        """
        self.pokemon_stats_path = pokemon_stats_path
        self._stats_data = None
        self._stats_dict = None
    
    def _load_stats_data(self) -> pd.DataFrame:
        """Load Pokemon stats data from CSV."""
        if self._stats_data is None:
            if not os.path.exists(self.pokemon_stats_path):
                raise FileNotFoundError(f"Pokemon stats file not found: {self.pokemon_stats_path}")
            
            self._stats_data = pd.read_csv(self.pokemon_stats_path)
            
            # Validate required columns
            required_cols = ['No', 'HP', 'atk', 'def', 'spa', 'spd', 'spe']
            missing_cols = [col for col in required_cols if col not in self._stats_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in Pokemon stats CSV: {missing_cols}")
        
        return self._stats_data
    
    def _build_stats_dict(self) -> Dict[int, np.ndarray]:
        """Build dictionary mapping Pokemon number to normalized base stats."""
        if self._stats_dict is None:
            data = self._load_stats_data()
            
            # Extract base stats columns (HP, Attack, Defense, Sp.Attack, Sp.Defense, Speed)
            stats_cols = ['HP', 'atk', 'def', 'spa', 'spd', 'spe']
            stats_array = data[stats_cols].values.astype(np.float32)
            
            # Normalize stats to 0-1 range
            # Pokemon base stats typically range from ~5 to 255
            # Using 255 as max for normalization
            normalized_stats = stats_array / 255.0
            
            # Clamp values to ensure they're in [0, 1] range
            normalized_stats = np.clip(normalized_stats, 0.0, 1.0)
            
            # Build dictionary: pokemon_no -> normalized_stats
            self._stats_dict = {}
            for idx, row in data.iterrows():
                pokemon_no = int(row['No'])
                self._stats_dict[pokemon_no] = normalized_stats[idx]
            
            # Add entry for unknown Pokemon (ID 0)
            self._stats_dict[0] = np.zeros(6, dtype=np.float32)
        
        return self._stats_dict
    
    def get_base_stats(self, pokemon_no: int) -> np.ndarray:
        """
        Get normalized base stats for a Pokemon.
        
        Args:
            pokemon_no: Pokemon number (0 for unknown)
        
        Returns:
            Array of 6 normalized base stats [HP, Atk, Def, SpA, SpD, Spe]
        """
        stats_dict = self._build_stats_dict()
        
        if pokemon_no not in stats_dict:
            # Return zeros for unknown Pokemon
            return np.zeros(6, dtype=np.float32)
        
        return stats_dict[pokemon_no].copy()
    
    def initialize_species_embeddings(
        self, 
        embedding_layer: nn.Embedding, 
        embed_dim: int = 32,
        freeze_base_stats: bool = False,
        random_init_std: float = 0.01
    ) -> None:
        """
        Initialize species embedding weights with base stats.
        
        Args:
            embedding_layer: PyTorch embedding layer to initialize
            embed_dim: Embedding dimension (should be >= 6)
            freeze_base_stats: Whether to freeze the first 6 dimensions during training
            random_init_std: Standard deviation for random initialization of remaining dimensions
        """
        if embed_dim < 6:
            raise ValueError(f"Embedding dimension must be >= 6, got {embed_dim}")
        
        vocab_size = embedding_layer.num_embeddings
        
        # Initialize with small random values
        nn.init.normal_(embedding_layer.weight, mean=0.0, std=random_init_std)
        
        # Set first 6 dimensions to normalized base stats
        stats_dict = self._build_stats_dict()
        
        with torch.no_grad():
            for pokemon_no, base_stats in stats_dict.items():
                if pokemon_no < vocab_size:
                    # Set first 6 dimensions to base stats
                    embedding_layer.weight[pokemon_no, :6] = torch.from_numpy(base_stats)
        
        # Optionally freeze base stats dimensions
        if freeze_base_stats:
            # Create a custom parameter that only updates the last (embed_dim - 6) dimensions
            # This is a bit complex, so for now we'll implement it as a warning
            # TODO: Implement proper freezing mechanism if needed
            print("Warning: freeze_base_stats=True is not yet fully implemented")
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size needed for embedding layer (max Pokemon number + 1)."""
        data = self._load_stats_data()
        max_no = data['No'].max()
        return max_no + 1  # +1 for 0-indexing
    
    def get_embedding_info(self) -> Dict[str, int]:
        """Get information about the embedding configuration."""
        vocab_size = self.get_vocab_size()
        return {
            'vocab_size': vocab_size,
            'pokemon_count': len(self._build_stats_dict()) - 1,  # -1 for unknown entry
            'embed_dim_minimum': 6,
            'recommended_embed_dim': 32
        }


def create_embedding_initializer(pokemon_stats_path: str = None) -> EmbeddingInitializer:
    """
    Factory function to create an embedding initializer.
    
    Args:
        pokemon_stats_path: Optional path to Pokemon stats CSV. If None, uses default.
    
    Returns:
        EmbeddingInitializer instance
    """
    if pokemon_stats_path is None:
        # Default path relative to project root
        pokemon_stats_path = "config/pokemon_stats.csv"
    
    return EmbeddingInitializer(pokemon_stats_path)