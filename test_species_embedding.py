#!/usr/bin/env python3
"""Quick test for Species Embedding functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.agents.species_embedding_layer import SpeciesEmbeddingLayer

def test_species_embedding_layer():
    """Test SpeciesEmbeddingLayer initialization and functionality."""
    print("🧪 Testing SpeciesEmbeddingLayer...")
    
    try:
        # Initialize layer
        layer = SpeciesEmbeddingLayer(
            vocab_size=1026,
            embed_dim=32,
            stats_csv_path="config/pokemon_stats.csv",
            device=torch.device("cpu")
        )
        print("✅ SpeciesEmbeddingLayer initialized successfully")
        
        # Test some known Pokemon IDs
        test_ids = [25, 6, 9, 792, 1024, 0]  # Pikachu, Charizard, Blastoise, Lunala, Terapagos, Unknown
        
        for pokemon_id in test_ids:
            embedding = layer.get_species_embedding(pokemon_id)
            print(f"Pokemon ID {pokemon_id}: embedding shape {embedding.shape}, sample values: {embedding[:3]}")
            
            # Check that embedding is normalized
            norm = torch.norm(embedding).item()
            print(f"  L2 norm: {norm:.4f} (should be close to 1.0)")
            
            # Check first 6 dimensions (should be base stats)
            stats_part = embedding[:6]
            print(f"  Base stats part: {stats_part}")
            
        print("✅ All embedding tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in SpeciesEmbeddingLayer test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_state_observer_integration():
    """Test StateObserver integration with species embeddings."""
    print("🧪 Testing StateObserver integration...")
    
    try:
        from src.state.state_observer import StateObserver
        
        # Initialize StateObserver
        observer = StateObserver("config/state_spec.yml")
        print("✅ StateObserver initialized successfully")
        
        # Test species embedding layer getter
        species_layer = observer._get_species_embedding_layer()
        if species_layer is not None:
            print("✅ Species embedding layer loaded successfully")
            
            # Test a simple embedding
            test_embedding = species_layer.get_species_embedding(25)  # Pikachu
            print(f"Test embedding shape: {test_embedding.shape}")
            print(f"Test embedding norm: {torch.norm(test_embedding).item():.4f}")
            
        else:
            print("❌ Species embedding layer failed to load")
            return False
            
        print("✅ StateObserver integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in StateObserver integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Species Embedding Phase 2 Tests")
    print("=" * 50)
    
    success = True
    
    success &= test_species_embedding_layer()
    print()
    success &= test_state_observer_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All Phase 2 tests PASSED! Species Embedding is working correctly.")
    else:
        print("❌ Some Phase 2 tests FAILED. Check the errors above.")
    
    sys.exit(0 if success else 1)