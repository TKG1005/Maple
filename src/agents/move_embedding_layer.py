"""
Move Embedding Layer for Neural Networks
Handles 256-dimensional move embeddings with learnable/non-learnable feature separation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle
from collections import OrderedDict


class MoveEmbeddingLayer(nn.Module):
    """
    Neural network layer for handling move embeddings with learnable/non-learnable features.
    
    This layer manages 256-dimensional move embeddings where:
    - 41 features are non-learnable (types, categories, scaled values, flags)
    - 215 features are learnable (description embeddings + additional parameters)
    """
    
    def __init__(self, embedding_file: str, device: torch.device = None):
        """
        Initialize the move embedding layer.
        
        Args:
            embedding_file: Path to the saved move embeddings file
            device: Device to place tensors on
        """
        super().__init__()
        
        self.device = device or torch.device('cpu')
        self.embedding_file = embedding_file
        
        # Load embeddings and metadata
        self.move_embeddings, self.feature_names, self.learnable_mask = self._load_embeddings()
        
        # Create move name to index mapping
        self.move_names = [name for name in self.move_embeddings.keys() if not name.startswith('move_')]
        self.move_to_idx = {name: idx for idx, name in enumerate(self.move_names)}
        
        # Separate learnable and non-learnable features
        # Use feature_names order as the canonical order
        self.learnable_indices = []
        self.non_learnable_indices = []
        
        for idx, feature_name in enumerate(self.feature_names):
            # Check mask using feature name (not relying on dict iteration order)
            if feature_name in self.learnable_mask and self.learnable_mask[feature_name]:
                self.learnable_indices.append(idx)
            else:
                self.non_learnable_indices.append(idx)
        
        # Create embedding matrix
        self.embedding_dim = len(self.feature_names)
        self.num_moves = len(self.move_names)
        
        # Initialize embedding matrix
        embedding_matrix = np.zeros((self.num_moves, self.embedding_dim), dtype=np.float32)
        for idx, move_name in enumerate(self.move_names):
            embedding_matrix[idx] = self.move_embeddings[move_name]
        
        # Split into learnable and non-learnable parts
        learnable_embeddings = embedding_matrix[:, self.learnable_indices]
        non_learnable_embeddings = embedding_matrix[:, self.non_learnable_indices]
        
        # Create learnable parameters
        self.learnable_embeddings = nn.Parameter(
            torch.from_numpy(learnable_embeddings).to(self.device),
            requires_grad=True
        )
        
        # Create non-learnable (frozen) parameters
        self.register_buffer(
            'non_learnable_embeddings',
            torch.from_numpy(non_learnable_embeddings).to(self.device)
        )
        
        print(f"MoveEmbeddingLayer initialized:")
        print(f"  - Total moves: {self.num_moves}")
        print(f"  - Embedding dimension: {self.embedding_dim}")
        print(f"  - Learnable features: {len(self.learnable_indices)}")
        print(f"  - Non-learnable features: {len(self.non_learnable_indices)}")
        print(f"  - Device: {self.device}")
    
    def _load_embeddings(self) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, bool]]:
        """Load embeddings from file."""
        with open(self.embedding_file, 'rb') as f:
            embedding_data = pickle.load(f)
        
        move_embeddings = embedding_data['move_embeddings']
        feature_names = embedding_data['feature_names']
        learnable_mask = embedding_data['learnable_mask']
        
        # Ensure learnable_mask is ordered consistently with feature_names
        if not isinstance(learnable_mask, OrderedDict):
            ordered_mask = OrderedDict()
            for feature_name in feature_names:
                if feature_name in learnable_mask:
                    ordered_mask[feature_name] = learnable_mask[feature_name]
                else:
                    # Default to False if missing (shouldn't happen)
                    print(f"Warning: Feature '{feature_name}' not found in learnable_mask, defaulting to False")
                    ordered_mask[feature_name] = False
            learnable_mask = ordered_mask
        
        return move_embeddings, feature_names, learnable_mask
    
    def forward(self, move_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get move embeddings using optimized index_select.
        
        Args:
            move_indices: Tensor of move indices [batch_size] or [batch_size, num_moves]
            
        Returns:
            Move embeddings [batch_size, embedding_dim] or [batch_size, num_moves, embedding_dim]
        """
        # Flatten indices for index_select
        original_shape = move_indices.shape
        flat_indices = move_indices.view(-1)
        
        # Use index_select for efficient gathering (single memory access)
        learnable_part = torch.index_select(self.learnable_embeddings, 0, flat_indices)
        non_learnable_part = torch.index_select(self.non_learnable_embeddings, 0, flat_indices)
        
        # Pre-allocate full embedding tensor
        batch_size = flat_indices.shape[0]
        full_embedding = torch.empty(batch_size, self.embedding_dim, device=self.device, dtype=torch.float32)
        
        # Use advanced indexing to fill both parts in one operation each
        full_embedding[:, self.learnable_indices] = learnable_part
        full_embedding[:, self.non_learnable_indices] = non_learnable_part
        
        # Reshape back to original shape + embedding dimension
        final_shape = original_shape + (self.embedding_dim,)
        return full_embedding.view(final_shape)
    
    def get_move_index(self, move_name: str) -> int:
        """Get the index for a move name."""
        return self.move_to_idx.get(move_name, 0)  # Default to first move if not found
    
    def get_move_embedding(self, move_name: str) -> torch.Tensor:
        """Get embedding for a specific move."""
        move_idx = self.get_move_index(move_name)
        return self.forward(torch.tensor([move_idx], device=self.device))[0]
    
    def get_learnable_parameters(self) -> torch.Tensor:
        """Get only the learnable parameters."""
        return self.learnable_embeddings
    
    def get_non_learnable_parameters(self) -> torch.Tensor:
        """Get only the non-learnable parameters (frozen)."""
        return self.non_learnable_embeddings
    
    def save_learned_embeddings(self, save_path: str):
        """Save the learned embeddings back to file."""
        # Reconstruct full embeddings
        full_embeddings = torch.zeros(self.num_moves, self.embedding_dim, device=self.device)
        full_embeddings[:, self.learnable_indices] = self.learnable_embeddings
        full_embeddings[:, self.non_learnable_indices] = self.non_learnable_embeddings
        
        # Convert back to numpy and update dictionary
        full_embeddings_np = full_embeddings.cpu().detach().numpy()
        updated_embeddings = {}
        
        for idx, move_name in enumerate(self.move_names):
            updated_embeddings[move_name] = full_embeddings_np[idx]
            updated_embeddings[f"move_{idx}"] = full_embeddings_np[idx]  # Also save with ID
        
        # Save to file
        embedding_data = {
            'move_embeddings': updated_embeddings,
            'feature_names': self.feature_names,
            'learnable_mask': self.learnable_mask,
            'feature_columns': self.feature_names,
            'embedding_dim': self.embedding_dim,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        print(f"Learned embeddings saved to {save_path}")
    
    def extra_repr(self) -> str:
        return f'num_moves={self.num_moves}, embedding_dim={self.embedding_dim}, ' \
               f'learnable={len(self.learnable_indices)}, ' \
               f'non_learnable={len(self.non_learnable_indices)}'


class MoveEmbeddingNetwork(nn.Module):
    """
    Example network using move embeddings for Pokemon move prediction/evaluation.
    """
    
    def __init__(self, embedding_file: str, hidden_dim: int = 256, 
                 output_dim: int = 4, device: torch.device = None):
        """
        Initialize network with move embeddings.
        
        Args:
            embedding_file: Path to move embeddings file
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (e.g., 4 for move selection)
            device: Device to use
        """
        super().__init__()
        
        self.device = device or torch.device('cpu')
        self.move_embedding_layer = MoveEmbeddingLayer(embedding_file, device)
        
        # Network layers
        self.hidden = nn.Linear(self.move_embedding_layer.embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Move to device
        self.to(self.device)
    
    def forward(self, move_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for move evaluation.
        
        Args:
            move_indices: Tensor of move indices [batch_size, 4] for 4 moves
            
        Returns:
            Move evaluations [batch_size, 4]
        """
        # Get move embeddings [batch_size, 4, embedding_dim]
        move_embeds = self.move_embedding_layer(move_indices)
        
        # Process each move
        batch_size, num_moves, embed_dim = move_embeds.shape
        move_embeds_flat = move_embeds.view(-1, embed_dim)  # [batch_size * 4, embedding_dim]
        
        # Pass through network
        hidden = self.activation(self.hidden(move_embeds_flat))
        hidden = self.dropout(hidden)
        output = self.output(hidden)  # [batch_size * 4, output_dim]
        
        # Reshape back to [batch_size, 4, output_dim]
        output = output.view(batch_size, num_moves, -1)
        
        # For move selection, we want [batch_size, 4] - one score per move
        if output.shape[-1] == 1:
            return output.squeeze(-1)
        else:
            return output
    
    def get_move_scores(self, move_names: List[str]) -> torch.Tensor:
        """Get scores for specific moves by name."""
        move_indices = [self.move_embedding_layer.get_move_index(name) for name in move_names]
        move_tensor = torch.tensor([move_indices], device=self.device)
        return self.forward(move_tensor)[0]


if __name__ == "__main__":
    # Example usage
    print("Testing MoveEmbeddingLayer...")
    
    # Test with 256-dimensional embeddings
    embedding_file = "config/move_embeddings_256d.pkl"
    if Path(embedding_file).exists():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create embedding layer
        embed_layer = MoveEmbeddingLayer(embedding_file, device)
        
        # Test forward pass
        test_indices = torch.tensor([0, 1, 2, 3], device=device)
        embeddings = embed_layer(test_indices)
        print(f"Test embeddings shape: {embeddings.shape}")
        
        # Test specific move
        move_embedding = embed_layer.get_move_embedding("はたく")
        print(f"'はたく' embedding shape: {move_embedding.shape}")
        print(f"'はたく' embedding range: [{move_embedding.min():.3f}, {move_embedding.max():.3f}]")
        
        # Test network
        network = MoveEmbeddingNetwork(embedding_file, device=device)
        move_scores = network.get_move_scores(["はたく", "かみなり", "つるぎのまい", "みがわり"])
        print(f"Move scores: {move_scores}")
        
        print("MoveEmbeddingLayer test completed successfully!")
    else:
        print(f"Embedding file {embedding_file} not found. Please run move embedding generation first.")