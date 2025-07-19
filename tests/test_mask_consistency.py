"""
Test module for verifying learnable_mask consistency between save and load.
"""

import pytest
import numpy as np
import tempfile
import os
from collections import OrderedDict
from src.utils.move_embedding import MoveEmbeddingGenerator
from src.agents.move_embedding_layer import MoveEmbeddingLayer


class TestMaskConsistency:
    """Test cases for learnable_mask ordering consistency."""
    
    def test_save_load_mask_consistency(self):
        """Test that learnable_mask maintains consistent ordering through save/load cycle."""
        # Create a minimal generator
        generator = MoveEmbeddingGenerator()
        
        # Create synthetic embeddings and mask
        num_features = 20
        feature_names = [f"feature_{i}" for i in range(num_features)]
        
        # Create learnable mask with specific pattern
        learnable_mask = OrderedDict()
        for i, name in enumerate(feature_names):
            # Make every 3rd feature learnable
            learnable_mask[name] = (i % 3 == 0)
        
        # Create dummy embeddings
        move_embeddings = {
            "move1": np.random.randn(num_features).astype(np.float32),
            "move2": np.random.randn(num_features).astype(np.float32),
            "move3": np.random.randn(num_features).astype(np.float32)
        }
        
        # Save embeddings
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_file = tmp.name
        
        try:
            generator.save_embeddings(move_embeddings, feature_names, temp_file, learnable_mask)
            
            # Load embeddings back
            loaded_embeddings, loaded_features, loaded_mask = generator.load_embeddings(temp_file)
            
            # Verify mask consistency
            assert isinstance(loaded_mask, OrderedDict), "Loaded mask should be OrderedDict"
            assert list(loaded_mask.keys()) == feature_names, "Mask keys should match feature names order"
            
            # Verify values are preserved
            for name in feature_names:
                assert loaded_mask[name] == learnable_mask[name], f"Mask value mismatch for {name}"
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_embedding_layer_indices_consistency(self):
        """Test that MoveEmbeddingLayer correctly identifies learnable indices."""
        # Create test data
        num_features = 10
        feature_names = [f"feature_{i}" for i in range(num_features)]
        
        # Create specific learnable pattern
        learnable_mask = OrderedDict()
        expected_learnable = []
        expected_non_learnable = []
        
        for i, name in enumerate(feature_names):
            is_learnable = (i < 5)  # First 5 are learnable
            learnable_mask[name] = is_learnable
            if is_learnable:
                expected_learnable.append(i)
            else:
                expected_non_learnable.append(i)
        
        # Create dummy embeddings
        move_embeddings = {
            "move1": np.ones(num_features).astype(np.float32)
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_file = tmp.name
        
        try:
            # Save using direct pickle to mimic MoveEmbeddingGenerator
            import pickle
            embedding_data = {
                'move_embeddings': move_embeddings,
                'feature_names': feature_names,
                'learnable_mask': learnable_mask,
                'embedding_dim': num_features
            }
            
            with open(temp_file, 'wb') as f:
                pickle.dump(embedding_data, f)
            
            # Load with MoveEmbeddingLayer
            layer = MoveEmbeddingLayer(temp_file)
            
            # Verify indices
            assert layer.learnable_indices == expected_learnable, \
                f"Learnable indices mismatch: {layer.learnable_indices} != {expected_learnable}"
            assert layer.non_learnable_indices == expected_non_learnable, \
                f"Non-learnable indices mismatch: {layer.non_learnable_indices} != {expected_non_learnable}"
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_legacy_dict_conversion(self):
        """Test that old dict-based masks are converted properly."""
        generator = MoveEmbeddingGenerator()
        
        # Create test data with regular dict (simulating old format)
        feature_names = ["feature_a", "feature_b", "feature_c", "feature_d"]
        old_mask = {
            "feature_d": True,
            "feature_a": False,
            "feature_c": True,
            "feature_b": False
        }
        
        move_embeddings = {
            "move1": np.array([1, 2, 3, 4], dtype=np.float32)
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_file = tmp.name
        
        try:
            # Save with old dict format
            import pickle
            embedding_data = {
                'move_embeddings': move_embeddings,
                'feature_names': feature_names,
                'learnable_mask': old_mask,  # Regular dict, not OrderedDict
                'embedding_dim': len(feature_names)
            }
            
            with open(temp_file, 'wb') as f:
                pickle.dump(embedding_data, f)
            
            # Load and verify conversion
            loaded_embeddings, loaded_features, loaded_mask = generator.load_embeddings(temp_file)
            
            # Should be converted to OrderedDict with feature_names order
            assert isinstance(loaded_mask, OrderedDict)
            assert list(loaded_mask.keys()) == feature_names
            
            # Values should be preserved
            assert loaded_mask["feature_a"] == False
            assert loaded_mask["feature_b"] == False
            assert loaded_mask["feature_c"] == True
            assert loaded_mask["feature_d"] == True
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_mask_ordering_stability(self):
        """Test that mask ordering remains stable across multiple save/load cycles."""
        generator = MoveEmbeddingGenerator()
        
        # Create complex feature names to test ordering
        feature_names = [
            "type_electric", "type_water", "category_Physical",
            "power_scaled", "accuracy_scaled", "desc_emb_0",
            "desc_emb_1", "learnable_0", "learnable_1", "learnable_2"
        ]
        
        # Create specific pattern
        learnable_mask = OrderedDict()
        for name in feature_names:
            learnable_mask[name] = name.startswith("learnable_") or name.startswith("desc_")
        
        move_embeddings = {
            "thunderbolt": np.random.randn(len(feature_names)).astype(np.float32)
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            temp_file = tmp.name
        
        try:
            # Multiple save/load cycles
            for cycle in range(3):
                generator.save_embeddings(move_embeddings, feature_names, temp_file, learnable_mask)
                loaded_embeddings, loaded_features, loaded_mask = generator.load_embeddings(temp_file)
                
                # Verify ordering is preserved
                assert list(loaded_mask.keys()) == feature_names, f"Order mismatch in cycle {cycle}"
                
                # Use loaded data for next cycle
                move_embeddings = loaded_embeddings
                feature_names = loaded_features
                learnable_mask = loaded_mask
                
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])