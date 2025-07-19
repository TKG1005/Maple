"""
Performance test for optimized MoveEmbeddingLayer.
Tests memory usage and speed improvements.
"""

import pytest
import torch
import time
import tempfile
import numpy as np
import pickle
from collections import OrderedDict
from src.agents.move_embedding_layer import MoveEmbeddingLayer


class TestMoveEmbeddingPerformance:
    """Performance tests for MoveEmbeddingLayer optimizations."""
    
    @pytest.fixture
    def create_test_embeddings(self):
        """Create test embedding file."""
        # Create realistic size embeddings (781 moves, 256 dimensions)
        num_moves = 781
        embedding_dim = 256
        num_learnable = 88
        num_non_learnable = 168
        
        # Create feature names and learnable mask
        feature_names = [f"feature_{i}" for i in range(embedding_dim)]
        learnable_mask = OrderedDict()
        for i, name in enumerate(feature_names):
            learnable_mask[name] = i >= num_non_learnable  # Last 88 are learnable
        
        # Create move embeddings with both name and move_id keys
        move_embeddings = {}
        move_names = []
        for i in range(num_moves):
            move_name = f"test_move_{i}"
            move_names.append(move_name)
            move_embeddings[move_name] = np.random.randn(embedding_dim).astype(np.float32)
            # Also add with move_id key
            move_embeddings[f"move_{i}"] = move_embeddings[move_name]
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_file = tmp.name
            
        embedding_data = {
            'move_embeddings': move_embeddings,
            'feature_names': feature_names,
            'learnable_mask': learnable_mask,
            'embedding_dim': embedding_dim
        }
        
        with open(temp_file, 'wb') as f:
            pickle.dump(embedding_data, f)
            
        yield temp_file
        
        # Cleanup
        import os
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
    def test_forward_pass_performance(self, create_test_embeddings):
        """Test forward pass performance with optimized index_select."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        layer = MoveEmbeddingLayer(create_test_embeddings, device)
        
        # Test different batch sizes
        batch_sizes = [1, 32, 128, 512]
        num_moves_per_sample = 4
        
        print("\nForward Pass Performance Test:")
        print(f"Device: {device}")
        
        for batch_size in batch_sizes:
            # Create random indices
            indices = torch.randint(0, layer.num_moves, (batch_size, num_moves_per_sample), device=device)
            
            # Warmup
            for _ in range(10):
                _ = layer(indices)
            
            # Time forward pass
            num_iterations = 100
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            for _ in range(num_iterations):
                embeddings = layer(indices)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed_time = time.time() - start_time
            
            avg_time_ms = (elapsed_time / num_iterations) * 1000
            print(f"Batch size {batch_size}: {avg_time_ms:.3f} ms per forward pass")
            
            # Verify output shape
            assert embeddings.shape == (batch_size, num_moves_per_sample, 256)
    
    def test_memory_efficiency(self, create_test_embeddings):
        """Test that the layer uses memory efficiently."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get initial memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated()
        
        # Create layer
        layer = MoveEmbeddingLayer(create_test_embeddings, device)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            layer_memory = torch.cuda.memory_allocated() - initial_memory
            memory_mb = layer_memory / (1024 * 1024)
            
            print(f"\nMemory Usage Test:")
            print(f"Layer memory usage: {memory_mb:.2f} MB")
            
            # Expected memory: (781 moves × 256 dims × 4 bytes/float32) ≈ 0.8 MB
            # Should not be doubled due to efficient storage
            expected_memory_mb = (layer.num_moves * 256 * 4) / (1024 * 1024)
            print(f"Expected memory usage: {expected_memory_mb:.2f} MB")
            
            # Allow some overhead but ensure it's not doubled
            assert memory_mb < expected_memory_mb * 1.5, "Memory usage too high, possible duplication"
    
    def test_index_select_correctness(self, create_test_embeddings):
        """Verify that index_select produces correct results."""
        device = torch.device('cpu')  # Use CPU for deterministic testing
        layer = MoveEmbeddingLayer(create_test_embeddings, device)
        
        # Test single index
        index = torch.tensor([5], device=device)
        embedding = layer(index)
        
        # Manually reconstruct expected embedding
        expected = torch.zeros(256)
        expected[layer.learnable_indices] = layer.learnable_embeddings[5]
        expected[layer.non_learnable_indices] = layer.non_learnable_embeddings[5]
        
        torch.testing.assert_close(embedding[0], expected, rtol=1e-5, atol=1e-5)
        
        # Test batch of indices
        indices = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        embeddings = layer(indices)
        
        assert embeddings.shape == (2, 3, 256)
        
        # Verify first move of first batch
        expected_first = torch.zeros(256)
        expected_first[layer.learnable_indices] = layer.learnable_embeddings[1]
        expected_first[layer.non_learnable_indices] = layer.non_learnable_embeddings[1]
        
        torch.testing.assert_close(embeddings[0, 0], expected_first, rtol=1e-5, atol=1e-5)
    
    def test_gradient_flow(self, create_test_embeddings):
        """Ensure gradients flow only through learnable parameters."""
        device = torch.device('cpu')
        layer = MoveEmbeddingLayer(create_test_embeddings, device)
        
        # Forward pass
        indices = torch.tensor([[0, 1], [2, 3]], device=device)
        embeddings = layer(indices)
        
        # Create dummy loss
        loss = embeddings.sum()
        loss.backward()
        
        # Check gradients
        assert layer.learnable_embeddings.grad is not None, "Learnable embeddings should have gradients"
        assert not hasattr(layer.non_learnable_embeddings, 'grad') or layer.non_learnable_embeddings.grad is None, \
            "Non-learnable embeddings should not have gradients"
        
        # Verify gradient shape
        assert layer.learnable_embeddings.grad.shape == layer.learnable_embeddings.shape
        
        print("\nGradient Flow Test:")
        print(f"Learnable parameters gradient norm: {layer.learnable_embeddings.grad.norm().item():.4f}")
        print("Non-learnable parameters have no gradients: ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])