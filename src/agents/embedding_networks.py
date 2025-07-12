"""
Pokemon Species Embedding Networks

This module provides neural network architectures that integrate Pokemon species embeddings
for improved tactical decision making in Pokemon battles.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
from .embedding_initializer import EmbeddingInitializer


class EmbeddingPolicyNetwork(nn.Module):
    """Policy network with Pokemon species embedding integration."""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_size: int = 128,
        use_2layer: bool = True,
        embedding_config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be gym.spaces.Box")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("action_space must be gym.spaces.Discrete")
        
        # Set default embedding configuration
        if embedding_config is None:
            embedding_config = {}
        
        self.embed_dim = embedding_config.get("embed_dim", 32)
        self.vocab_size = embedding_config.get("vocab_size", 1026)  # 0 + 1025 Pokemon
        self.freeze_base_stats = embedding_config.get("freeze_base_stats", False)
        
        # Species indices in the state vector (based on StateObserver analysis)
        # Indices 836-847: my_team[0-5].species_id + opp_team[0-5].species_id
        self.species_indices = embedding_config.get(
            "species_indices", 
            list(range(836, 848))  # [836, 837, ..., 847]
        )
        self.num_species_features = len(self.species_indices)
        
        # Original observation dimension
        obs_dim = int(observation_space.shape[0])
        action_dim = int(action_space.n)
        
        # Create species embedding layer
        self.species_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim
        )
        
        # Initialize embeddings with base stats
        self._initialize_embeddings()
        
        # Calculate dimensions for the main network
        # Remove the 12 species_id features and add 12 * embed_dim features
        non_species_dim = obs_dim - self.num_species_features
        embedded_species_dim = self.num_species_features * self.embed_dim
        total_input_dim = non_species_dim + embedded_species_dim
        
        # Main policy network
        if use_2layer:
            self.main_network = nn.Sequential(
                nn.Linear(total_input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
            )
        else:
            self.main_network = nn.Sequential(
                nn.Linear(total_input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
            )
    
    def _initialize_embeddings(self) -> None:
        """Initialize species embeddings with base stats."""
        try:
            initializer = EmbeddingInitializer()
            initializer.initialize_species_embeddings(
                embedding_layer=self.species_embedding,
                embed_dim=self.embed_dim,
                freeze_base_stats=self.freeze_base_stats
            )
        except Exception as e:
            print(f"Warning: Could not initialize species embeddings with base stats: {e}")
            # Fallback to default random initialization
            nn.init.normal_(self.species_embedding.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with species embedding integration.
        
        Args:
            x: State tensor of shape [batch_size, obs_dim] or [obs_dim]
        
        Returns:
            Action logits tensor
        """
        # Handle both batch and single observation inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.size(0)
        
        # Extract species IDs (convert to long for embedding lookup)
        species_ids = x[:, self.species_indices].long()  # [batch_size, num_species_features]
        
        # Get embeddings for all species
        species_embeddings = self.species_embedding(species_ids)  # [batch_size, num_species_features, embed_dim]
        
        # Flatten species embeddings
        species_embeddings_flat = species_embeddings.view(batch_size, -1)  # [batch_size, num_species_features * embed_dim]
        
        # Extract non-species features
        non_species_features = self._extract_non_species_features(x)  # [batch_size, non_species_dim]
        
        # Concatenate non-species features with embedded species features
        combined_features = torch.cat([non_species_features, species_embeddings_flat], dim=1)
        
        # Pass through main network
        output = self.main_network(combined_features)
        
        # Squeeze if input was 1D
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
    
    def _extract_non_species_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract all features except species IDs from the state vector."""
        # Create a mask for non-species indices
        all_indices = torch.arange(x.size(1), device=x.device)
        species_indices_tensor = torch.tensor(self.species_indices, device=x.device)
        
        # Create boolean mask: True for non-species indices
        mask = ~torch.isin(all_indices, species_indices_tensor)
        
        # Extract non-species features
        return x[:, mask]


class EmbeddingValueNetwork(nn.Module):
    """Value network with Pokemon species embedding integration."""
    
    def __init__(
        self,
        observation_space: gym.Space,
        hidden_size: int = 128,
        use_2layer: bool = True,
        embedding_config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__()
        
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be gym.spaces.Box")
        
        # Set default embedding configuration
        if embedding_config is None:
            embedding_config = {}
        
        self.embed_dim = embedding_config.get("embed_dim", 32)
        self.vocab_size = embedding_config.get("vocab_size", 1026)  # 0 + 1025 Pokemon
        self.freeze_base_stats = embedding_config.get("freeze_base_stats", False)
        
        # Species indices in the state vector
        self.species_indices = embedding_config.get(
            "species_indices", 
            list(range(836, 848))  # [836, 837, ..., 847]
        )
        self.num_species_features = len(self.species_indices)
        
        # Original observation dimension
        obs_dim = int(observation_space.shape[0])
        
        # Create species embedding layer
        self.species_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim
        )
        
        # Initialize embeddings with base stats
        self._initialize_embeddings()
        
        # Calculate dimensions for the main network
        non_species_dim = obs_dim - self.num_species_features
        embedded_species_dim = self.num_species_features * self.embed_dim
        total_input_dim = non_species_dim + embedded_species_dim
        
        # Main value network
        if use_2layer:
            self.main_network = nn.Sequential(
                nn.Linear(total_input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.main_network = nn.Sequential(
                nn.Linear(total_input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
    
    def _initialize_embeddings(self) -> None:
        """Initialize species embeddings with base stats."""
        try:
            initializer = EmbeddingInitializer()
            initializer.initialize_species_embeddings(
                embedding_layer=self.species_embedding,
                embed_dim=self.embed_dim,
                freeze_base_stats=self.freeze_base_stats
            )
        except Exception as e:
            print(f"Warning: Could not initialize species embeddings with base stats: {e}")
            # Fallback to default random initialization
            nn.init.normal_(self.species_embedding.weight, mean=0.0, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with species embedding integration.
        
        Args:
            x: State tensor of shape [batch_size, obs_dim] or [obs_dim]
        
        Returns:
            Value tensor (squeezed to remove last dimension)
        """
        # Handle both batch and single observation inputs
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.size(0)
        
        # Extract species IDs (convert to long for embedding lookup)
        species_ids = x[:, self.species_indices].long()  # [batch_size, num_species_features]
        
        # Get embeddings for all species
        species_embeddings = self.species_embedding(species_ids)  # [batch_size, num_species_features, embed_dim]
        
        # Flatten species embeddings
        species_embeddings_flat = species_embeddings.view(batch_size, -1)  # [batch_size, num_species_features * embed_dim]
        
        # Extract non-species features
        non_species_features = self._extract_non_species_features(x)  # [batch_size, non_species_dim]
        
        # Concatenate non-species features with embedded species features
        combined_features = torch.cat([non_species_features, species_embeddings_flat], dim=1)
        
        # Pass through main network
        output = self.main_network(combined_features).squeeze(-1)
        
        # Squeeze if input was 1D
        if squeeze_output:
            output = output.squeeze(0)
        
        return output
    
    def _extract_non_species_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract all features except species IDs from the state vector."""
        # Create a mask for non-species indices
        all_indices = torch.arange(x.size(1), device=x.device)
        species_indices_tensor = torch.tensor(self.species_indices, device=x.device)
        
        # Create boolean mask: True for non-species indices
        mask = ~torch.isin(all_indices, species_indices_tensor)
        
        # Extract non-species features
        return x[:, mask]


def get_embedding_network_info(network) -> Dict[str, Any]:
    """Get information about an embedding network instance.
    
    Args:
        network: EmbeddingPolicyNetwork or EmbeddingValueNetwork instance
        
    Returns:
        Dictionary with network information including embedding details
    """
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    info = {
        "type": network.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "has_species_embedding": True,
        "embedding_vocab_size": network.vocab_size,
        "embedding_dim": network.embed_dim,
        "num_species_features": network.num_species_features,
        "species_indices": network.species_indices,
        "freeze_base_stats": network.freeze_base_stats
    }
    
    # Add embedding layer specific information
    embedding_params = sum(p.numel() for p in network.species_embedding.parameters())
    info["embedding_params"] = embedding_params
    info["embedding_trainable"] = sum(p.numel() for p in network.species_embedding.parameters() if p.requires_grad)
    
    return info