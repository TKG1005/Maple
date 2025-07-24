"""Species Embedding Layer for Pokemon state space normalization.

This module provides embedding functionality for Pokemon species IDs,
converting raw Pokedex numbers into normalized dense embeddings initialized
with base stats.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpeciesEmbeddingLayer(nn.Module):
    """Embedding layer for Pokemon species with base stats initialization.
    
    This layer converts Pokemon species IDs (Pokedex numbers) into dense embeddings,
    initialized with normalized base stats for better learning convergence.
    """
    
    def __init__(
        self,
        vocab_size: int = 1026,  # 1025 Pokemon + 1 unknown
        embed_dim: int = 32,
        stats_csv_path: str = "config/pokemon_stats.csv",
        device: Optional[torch.device] = None,
    ):
        """Initialize species embedding layer.
        
        Args:
            vocab_size: Number of Pokemon species + 1 for unknown (default: 1026)
            embed_dim: Embedding dimension (default: 32)
            stats_csv_path: Path to Pokemon stats CSV file
            device: Torch device for embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.device = device or torch.device("cpu")
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, device=self.device)
        
        # Initialize with base stats
        self._init_with_base_stats(stats_csv_path)
        
    def _init_with_base_stats(self, stats_csv_path: str):
        """Initialize embedding weights with normalized base stats.
        
        The first 6 dimensions are initialized with normalized base stats
        (HP, Attack, Defense, Sp.Attack, Sp.Defense, Speed).
        Remaining dimensions are initialized with small random values.
        """
        try:
            # Load Pokemon stats
            df = pd.read_csv(stats_csv_path)
            logger.info(f"Loaded {len(df)} Pokemon from {stats_csv_path}")
            
            # Initialize embedding weights
            with torch.no_grad():
                # Initialize all weights with small random values
                nn.init.xavier_uniform_(self.embedding.weight, gain=0.1)
                
                # Override first 6 dimensions with normalized base stats
                for _, row in df.iterrows():
                    pokedex_num = int(row["pokedex_number"])
                    if pokedex_num < self.vocab_size:
                        # Normalize base stats to [0, 1] range
                        stats = torch.tensor([
                            row["hp"] / 255.0,           # HP max: 255
                            row["attack"] / 255.0,        # Attack max: 255
                            row["defense"] / 255.0,       # Defense max: 255
                            row["sp_attack"] / 255.0,     # Sp.Attack max: 255
                            row["sp_defense"] / 255.0,    # Sp.Defense max: 255
                            row["speed"] / 255.0,         # Speed max: 255
                        ], dtype=torch.float32, device=self.device)
                        
                        # Set first 6 dimensions to normalized stats
                        self.embedding.weight[pokedex_num, :6] = stats
                        
                # Set unknown Pokemon (ID 0) to zeros
                self.embedding.weight[0] = 0.0
                
            logger.info(f"Initialized embeddings with base stats for {len(df)} Pokemon")
            
        except Exception as e:
            logger.warning(f"Failed to load base stats: {e}. Using random initialization.")
            
    def forward(self, species_ids: torch.Tensor) -> torch.Tensor:
        """Convert species IDs to normalized embeddings.
        
        Args:
            species_ids: Tensor of species IDs (Pokedex numbers)
            
        Returns:
            Normalized embeddings with shape (..., embed_dim)
        """
        # Ensure IDs are within valid range
        species_ids = species_ids.long()
        species_ids = torch.clamp(species_ids, 0, self.vocab_size - 1)
        
        # Get embeddings
        embeddings = self.embedding(species_ids)
        
        # L2 normalize embeddings for stable scale
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def get_species_embedding(self, species_id: int) -> torch.Tensor:
        """Get embedding for a single species ID.
        
        Args:
            species_id: Pokemon species ID (Pokedex number)
            
        Returns:
            Normalized embedding vector
        """
        species_tensor = torch.tensor([species_id], dtype=torch.long, device=self.device)
        return self.forward(species_tensor).squeeze(0)