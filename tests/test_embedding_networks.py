"""Tests for Pokemon species embedding networks."""

import pytest
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import tempfile
import os

from src.agents.embedding_initializer import EmbeddingInitializer, create_embedding_initializer
from src.agents.embedding_networks import EmbeddingPolicyNetwork, EmbeddingValueNetwork, get_embedding_network_info
from src.agents.network_factory import create_policy_network, create_value_network, get_network_info


class TestEmbeddingInitializer:
    """Test the embedding initializer functionality."""
    
    def test_initialization_with_default_path(self):
        """Test initializer with default Pokemon stats path."""
        initializer = EmbeddingInitializer()
        assert initializer.pokemon_stats_path == "config/pokemon_stats.csv"
    
    def test_get_base_stats_known_pokemon(self):
        """Test getting base stats for known Pokemon."""
        initializer = create_embedding_initializer()
        
        # Test Pikachu (No. 25) - should have non-zero stats
        pikachu_stats = initializer.get_base_stats(25)
        assert pikachu_stats.shape == (6,)
        assert np.all(pikachu_stats >= 0.0)
        assert np.all(pikachu_stats <= 1.0)
        assert np.sum(pikachu_stats) > 0  # Should not be all zeros
    
    def test_get_base_stats_unknown_pokemon(self):
        """Test getting base stats for unknown Pokemon (ID 0)."""
        initializer = create_embedding_initializer()
        
        unknown_stats = initializer.get_base_stats(0)
        assert unknown_stats.shape == (6,)
        assert np.all(unknown_stats == 0.0)
    
    def test_get_base_stats_nonexistent_pokemon(self):
        """Test getting base stats for non-existent Pokemon ID."""
        initializer = create_embedding_initializer()
        
        # Test with ID that doesn't exist
        nonexistent_stats = initializer.get_base_stats(9999)
        assert nonexistent_stats.shape == (6,)
        assert np.all(nonexistent_stats == 0.0)
    
    def test_initialize_species_embeddings(self):
        """Test initializing embedding layer with base stats."""
        initializer = create_embedding_initializer()
        
        # Create embedding layer
        vocab_size = 100  # Small for testing
        embed_dim = 32
        embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Initialize with base stats
        initializer.initialize_species_embeddings(embedding, embed_dim)
        
        # Check that embedding weights are initialized
        assert embedding.weight.shape == (vocab_size, embed_dim)
        
        # Check that first 6 dimensions contain base stats for known Pokemon
        if vocab_size > 25:  # If Pikachu is in vocab
            pikachu_embedding = embedding.weight[25, :6]
            expected_stats = torch.from_numpy(initializer.get_base_stats(25))
            torch.testing.assert_close(pikachu_embedding, expected_stats, atol=1e-5, rtol=1e-5)
    
    def test_get_vocab_size(self):
        """Test getting vocabulary size."""
        initializer = create_embedding_initializer()
        vocab_size = initializer.get_vocab_size()
        assert vocab_size > 1025  # Should be at least 1026 (0 + 1025 Pokemon)
    
    def test_get_embedding_info(self):
        """Test getting embedding configuration info."""
        initializer = create_embedding_initializer()
        info = initializer.get_embedding_info()
        
        assert "vocab_size" in info
        assert "pokemon_count" in info
        assert "embed_dim_minimum" in info
        assert "recommended_embed_dim" in info
        assert info["embed_dim_minimum"] == 6
        assert info["recommended_embed_dim"] == 32


class TestEmbeddingNetworks:
    """Test embedding policy and value networks."""
    
    @pytest.fixture
    def sample_spaces(self):
        """Create sample observation and action spaces."""
        obs_space = gym.spaces.Box(low=0, high=1, shape=(1136,), dtype=np.float32)
        action_space = gym.spaces.Discrete(10)
        return obs_space, action_space
    
    def test_embedding_policy_network_creation(self, sample_spaces):
        """Test creating an embedding policy network."""
        obs_space, action_space = sample_spaces
        
        embedding_config = {
            "embed_dim": 32,
            "vocab_size": 1026,
            "species_indices": list(range(836, 848))
        }
        
        network = EmbeddingPolicyNetwork(
            observation_space=obs_space,
            action_space=action_space,
            hidden_size=128,
            embedding_config=embedding_config
        )
        
        assert network.embed_dim == 32
        assert network.vocab_size == 1026
        assert len(network.species_indices) == 12
    
    def test_embedding_value_network_creation(self, sample_spaces):
        """Test creating an embedding value network."""
        obs_space, action_space = sample_spaces
        
        embedding_config = {
            "embed_dim": 32,
            "vocab_size": 1026,
            "species_indices": list(range(836, 848))
        }
        
        network = EmbeddingValueNetwork(
            observation_space=obs_space,
            hidden_size=128,
            embedding_config=embedding_config
        )
        
        assert network.embed_dim == 32
        assert network.vocab_size == 1026
        assert len(network.species_indices) == 12
    
    def test_policy_network_forward_pass(self, sample_spaces):
        """Test forward pass through embedding policy network."""
        obs_space, action_space = sample_spaces
        
        network = EmbeddingPolicyNetwork(
            observation_space=obs_space,
            action_space=action_space,
            hidden_size=64,
            embedding_config={"embed_dim": 16}
        )
        
        # Create sample observation with species IDs
        obs = torch.zeros(1136)
        # Set species IDs at correct indices (836-847)
        obs[836:848] = torch.tensor([25, 6, 9, 0, 0, 0, 150, 149, 144, 0, 0, 0])  # Sample Pokemon IDs
        
        # Forward pass
        output = network(obs)
        
        assert output.shape == (action_space.n,)
        assert not torch.isnan(output).any()
    
    def test_value_network_forward_pass(self, sample_spaces):
        """Test forward pass through embedding value network."""
        obs_space, action_space = sample_spaces
        
        network = EmbeddingValueNetwork(
            observation_space=obs_space,
            hidden_size=64,
            embedding_config={"embed_dim": 16}
        )
        
        # Create sample observation with species IDs
        obs = torch.zeros(1136)
        obs[836:848] = torch.tensor([25, 6, 9, 0, 0, 0, 150, 149, 144, 0, 0, 0])
        
        # Forward pass
        output = network(obs)
        
        assert output.shape == ()  # Scalar output
        assert not torch.isnan(output).any()
    
    def test_batch_forward_pass(self, sample_spaces):
        """Test forward pass with batch input."""
        obs_space, action_space = sample_spaces
        
        network = EmbeddingPolicyNetwork(
            observation_space=obs_space,
            action_space=action_space,
            hidden_size=64,
            embedding_config={"embed_dim": 16}
        )
        
        # Create batch of observations
        batch_size = 4
        obs_batch = torch.zeros(batch_size, 1136)
        for i in range(batch_size):
            obs_batch[i, 836:848] = torch.tensor([25, 6, 9, 0, 0, 0, 150, 149, 144, 0, 0, 0])
        
        # Forward pass
        output = network(obs_batch)
        
        assert output.shape == (batch_size, action_space.n)
        assert not torch.isnan(output).any()
    
    def test_get_embedding_network_info(self, sample_spaces):
        """Test getting information about embedding networks."""
        obs_space, action_space = sample_spaces
        
        network = EmbeddingPolicyNetwork(
            observation_space=obs_space,
            action_space=action_space,
            embedding_config={"embed_dim": 32}
        )
        
        info = get_embedding_network_info(network)
        
        assert "type" in info
        assert "has_species_embedding" in info
        assert "embedding_vocab_size" in info
        assert "embedding_dim" in info
        assert "num_species_features" in info
        assert "species_indices" in info
        assert info["has_species_embedding"] is True
        assert info["embedding_dim"] == 32


class TestNetworkFactory:
    """Test network factory integration with embedding networks."""
    
    @pytest.fixture
    def sample_spaces(self):
        """Create sample observation and action spaces."""
        obs_space = gym.spaces.Box(low=0, high=1, shape=(1136,), dtype=np.float32)
        action_space = gym.spaces.Discrete(10)
        return obs_space, action_space
    
    def test_create_embedding_policy_network(self, sample_spaces):
        """Test creating embedding policy network through factory."""
        obs_space, action_space = sample_spaces
        
        config = {
            "type": "embedding",
            "hidden_size": 128,
            "embedding_config": {
                "embed_dim": 32,
                "vocab_size": 1026
            }
        }
        
        network = create_policy_network(obs_space, action_space, config)
        
        assert isinstance(network, EmbeddingPolicyNetwork)
        assert network.embed_dim == 32
        assert network.vocab_size == 1026
    
    def test_create_embedding_value_network(self, sample_spaces):
        """Test creating embedding value network through factory."""
        obs_space, action_space = sample_spaces
        
        config = {
            "type": "embedding",
            "hidden_size": 128,
            "embedding_config": {
                "embed_dim": 32,
                "vocab_size": 1026
            }
        }
        
        network = create_value_network(obs_space, config)
        
        assert isinstance(network, EmbeddingValueNetwork)
        assert network.embed_dim == 32
        assert network.vocab_size == 1026
    
    def test_get_network_info_embedding(self, sample_spaces):
        """Test getting network info for embedding networks."""
        obs_space, action_space = sample_spaces
        
        config = {
            "type": "embedding",
            "embedding_config": {"embed_dim": 32}
        }
        
        network = create_policy_network(obs_space, action_space, config)
        info = get_network_info(network)
        
        assert info["type"] == "EmbeddingPolicyNetwork"
        assert info["has_species_embedding"] is True
        assert "embedding_params" in info


@pytest.mark.slow
class TestEmbeddingIntegration:
    """Integration tests for embedding networks (marked as slow)."""
    
    def test_embedding_network_training_step(self):
        """Test that embedding networks can perform a training step."""
        # This would test integration with actual training loop
        # For now, just verify that gradients flow correctly
        
        obs_space = gym.spaces.Box(low=0, high=1, shape=(1136,), dtype=np.float32)
        action_space = gym.spaces.Discrete(10)
        
        policy_net = EmbeddingPolicyNetwork(
            observation_space=obs_space,
            action_space=action_space,
            embedding_config={"embed_dim": 16}
        )
        
        # Create sample batch
        obs = torch.zeros(4, 1136)
        obs[:, 836:848] = torch.randint(0, 100, (4, 12))  # Random species IDs
        
        # Forward pass
        logits = policy_net(obs)
        loss = torch.mean(logits)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert policy_net.species_embedding.weight.grad is not None
        for param in policy_net.main_network.parameters():
            if param.requires_grad:
                assert param.grad is not None