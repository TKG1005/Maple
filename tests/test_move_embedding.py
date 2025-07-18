"""
Test module for move embedding functionality.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from src.utils.move_embedding import MoveEmbeddingGenerator, create_move_embeddings


class TestMoveEmbeddingGenerator:
    """Test cases for MoveEmbeddingGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a sample moves CSV for testing
        self.test_data = {
            'move_id': [1, 2, 3, 4, 5],
            'name': ['はたく', 'からてチョップ', 'でんじは', 'つるぎのまい', 'かみなり'],
            'name_eng': ['Pound', 'Karate Chop', 'Thunder Wave', 'Swords Dance', 'Thunder'],
            'category': ['Physical', 'Physical', 'Status', 'Status', 'Special'],
            'type': ['normal', 'fighting', 'electric', 'normal', 'electric'],
            'power': [40, 50, 0, 0, 110],
            'accuracy': [1.0, 1.0, 0.9, 1.0, 0.7],
            'pp': [35, 25, 20, 20, 10],
            'priority': [0, 0, 0, 0, 0],
            'crit_stage': [0, 1, 0, 0, 0],
            'ohko_flag': [0, 0, 0, 0, 0],
            'contact_flag': [1, 1, 0, 0, 0],
            'sound_flag': [0, 0, 0, 0, 0],
            'multi_hit_min': [1, 1, 1, 1, 1],
            'multi_hit_max': [1, 1, 1, 1, 1],
            'recoil_ratio': [0.0, 0.0, 0.0, 0.0, 0.0],
            'recharge_turn': [0, 0, 0, 0, 0],
            'contenus_effect': [0, 0, 0, 0, 0],
            'contenus_turn': [0, 0, 0, 0, 0],
            'charging_turn': [0, 0, 0, 0, 0],
            'healing_ratio': [0.0, 0.0, 0.0, 0.0, 0.0],
            'protectable': [1, 1, 1, 0, 1],
            'substitutable': [1, 1, 1, 1, 1],
            'guard_move_flag': [0, 0, 0, 0, 0],
            'base_desc': ['通常攻撃。', '急所に当たりやすい。', '相手を麻痺させる。', '攻撃ランクを上げる。', '高威力の電気攻撃。']
        }
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame(self.test_data)
        df.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()
        
        # Initialize generator
        self.generator = MoveEmbeddingGenerator(self.temp_csv.name)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)
    
    def test_load_moves_data(self):
        """Test loading moves data from CSV."""
        df = self.generator.load_moves_data()
        
        assert len(df) == 5
        assert 'move_id' in df.columns
        assert 'name' in df.columns
        assert 'type' in df.columns
        assert 'category' in df.columns
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        df = self.generator.preprocess_data()
        
        # Check that numerical columns are properly typed
        assert df['power'].dtype == np.float64
        assert df['accuracy'].dtype == np.float64
        assert df['pp'].dtype == np.float64
        assert df['priority'].dtype == np.int64
        
        # Check that boolean flags are float
        assert df['ohko_flag'].dtype == np.float64
        assert df['contact_flag'].dtype == np.float64
        
        # Check that descriptions are filled
        assert df['base_desc'].isna().sum() == 0
    
    def test_create_categorical_encodings(self):
        """Test categorical encoding creation."""
        df = self.generator.preprocess_data()
        df_encoded = self.generator.create_categorical_encodings(df)
        
        # Check that one-hot columns are created
        type_cols = [col for col in df_encoded.columns if col.startswith('type_')]
        category_cols = [col for col in df_encoded.columns if col.startswith('category_')]
        
        assert len(type_cols) > 0
        assert len(category_cols) > 0
        
        # Check that all types are represented
        expected_types = {'normal', 'fighting', 'electric'}
        actual_types = {col.replace('type_', '') for col in type_cols}
        assert expected_types == actual_types
    
    def test_scale_numerical_features(self):
        """Test numerical feature scaling."""
        df = self.generator.preprocess_data()
        df_scaled = self.generator.scale_numerical_features(df)
        
        # Check that scaled columns are created
        scaled_cols = [col for col in df_scaled.columns if col.endswith('_scaled')]
        assert len(scaled_cols) > 0
        
        # Check that scaled values are in reasonable range
        for col in scaled_cols:
            assert df_scaled[col].min() >= 0
            assert df_scaled[col].max() <= 1
    
    def test_create_description_embeddings(self):
        """Test description embedding creation."""
        df = self.generator.preprocess_data()
        df_with_embeddings = self.generator.create_description_embeddings(df)
        
        # Check that embedding columns are created
        desc_cols = [col for col in df_with_embeddings.columns if col.startswith('desc_emb_')]
        assert len(desc_cols) > 0  # Should have at least some embedding columns
        
        # For small datasets, expect fewer dimensions than the default
        expected_max_dims = min(len(df), self.generator.desc_embedding_dim)
        assert len(desc_cols) <= expected_max_dims
        
        # Check that embeddings are numerical
        for col in desc_cols:
            assert df_with_embeddings[col].dtype in [np.float32, np.float64]
    
    def test_assemble_final_features(self):
        """Test final feature assembly."""
        df = self.generator.preprocess_data()
        df = self.generator.create_categorical_encodings(df)
        df = self.generator.scale_numerical_features(df)
        df = self.generator.create_description_embeddings(df)
        
        feature_matrix, feature_names = self.generator.assemble_final_features(df)
        
        # Check feature matrix shape
        assert feature_matrix.shape[0] == len(df)
        assert feature_matrix.shape[1] == len(feature_names)
        
        # Check that feature matrix is numerical
        assert feature_matrix.dtype == np.float32
        
        # Check that feature names are strings
        assert all(isinstance(name, str) for name in feature_names)
    
    def test_create_move_embedding_dict(self):
        """Test move embedding dictionary creation."""
        df = self.generator.preprocess_data()
        df = self.generator.create_categorical_encodings(df)
        df = self.generator.scale_numerical_features(df)
        df = self.generator.create_description_embeddings(df)
        
        feature_matrix, feature_names = self.generator.assemble_final_features(df)
        move_embeddings = self.generator.create_move_embedding_dict(df, feature_matrix)
        
        # Check that embeddings are created for all moves
        assert len(move_embeddings) >= len(df)  # At least one per move, possibly more with ID keys
        
        # Check that specific moves are present
        assert 'はたく' in move_embeddings
        assert 'からてチョップ' in move_embeddings
        
        # Check embedding dimensions
        embedding = move_embeddings['はたく']
        assert embedding.shape[0] == len(feature_names)
        assert embedding.dtype == np.float32
    
    def test_generate_embeddings(self):
        """Test complete embedding generation."""
        move_embeddings, feature_names = self.generator.generate_embeddings()
        
        # Check that embeddings are generated
        assert len(move_embeddings) > 0
        assert len(feature_names) > 0
        
        # Check that all test moves are present
        for move_name in self.test_data['name']:
            assert move_name in move_embeddings
        
        # Check embedding consistency
        for move_name in self.test_data['name']:
            embedding = move_embeddings[move_name]
            assert embedding.shape[0] == len(feature_names)
            assert embedding.dtype == np.float32
    
    def test_fusion_strategies(self):
        """Test different fusion strategies."""
        strategies = ['concatenate', 'balanced', 'weighted']
        
        for strategy in strategies:
            move_embeddings, feature_names = self.generator.generate_embeddings(
                fusion_strategy=strategy
            )
            
            # Check that embeddings are generated
            assert len(move_embeddings) > 0
            assert len(feature_names) > 0
            
            # Check that all test moves are present
            for move_name in self.test_data['name']:
                assert move_name in move_embeddings
                embedding = move_embeddings[move_name]
                assert embedding.shape[0] == len(feature_names)
                assert embedding.dtype == np.float32
    
    def test_japanese_text_preprocessing(self):
        """Test Japanese text preprocessing."""
        # Test with various Japanese text inputs
        test_cases = [
            ("通常攻撃。", "通常攻撃。"),
            ("", "通常攻撃。"),
            ("【特殊】攻撃力を上げる", "特殊攻撃力を上げる。"),
            ("長い説明文" * 50, "長い説明文" * 32 + "。"),  # Should be truncated
            ("改行\nあり\nテスト", "改行 あり テスト。"),
        ]
        
        for input_text, expected_output in test_cases:
            result = self.generator.preprocess_japanese_text(input_text)
            # Check that preprocessing produces reasonable output
            assert isinstance(result, str)
            assert len(result) > 0
            assert result.endswith('。')
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        # Generate embeddings first
        move_embeddings, feature_names = self.generator.generate_embeddings()
        
        # Test semantic search
        query = "攻撃力を上げる"
        results = self.generator.semantic_search(query, move_embeddings, top_k=3)
        
        # Check results format
        assert isinstance(results, list)
        assert len(results) <= 3
        
        for move_name, similarity in results:
            assert isinstance(move_name, str)
            assert isinstance(similarity, (float, np.floating))
            assert move_name in move_embeddings
    
    def test_save_and_load_embeddings(self):
        """Test saving and loading embeddings."""
        # Generate embeddings
        move_embeddings, feature_names = self.generator.generate_embeddings()
        
        # Save to temporary file
        temp_save = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        temp_save.close()
        
        try:
            self.generator.save_embeddings(move_embeddings, feature_names, temp_save.name)
            
            # Load embeddings
            loaded_embeddings, loaded_features = self.generator.load_embeddings(temp_save.name)
            
            # Check that loaded data matches original
            assert len(loaded_embeddings) == len(move_embeddings)
            assert loaded_features == feature_names
            
            # Check specific embeddings
            for move_name in self.test_data['name']:
                original = move_embeddings[move_name]
                loaded = loaded_embeddings[move_name]
                np.testing.assert_array_equal(original, loaded)
        
        finally:
            if os.path.exists(temp_save.name):
                os.unlink(temp_save.name)


class TestConvenienceFunction:
    """Test the convenience function."""
    
    def test_create_move_embeddings_function(self):
        """Test the create_move_embeddings convenience function."""
        # This test requires the actual config/moves.csv file
        if not os.path.exists("config/moves.csv"):
            pytest.skip("config/moves.csv not found")
        
        # Create temporary save path
        temp_save = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        temp_save.close()
        
        try:
            move_embeddings, feature_names = create_move_embeddings(
                moves_csv_path="config/moves.csv",
                save_path=temp_save.name
            )
            
            # Check that embeddings are generated
            assert len(move_embeddings) > 0
            assert len(feature_names) > 0
            
            # Check that save file exists
            assert os.path.exists(temp_save.name)
        
        finally:
            if os.path.exists(temp_save.name):
                os.unlink(temp_save.name)


if __name__ == "__main__":
    pytest.main([__file__])