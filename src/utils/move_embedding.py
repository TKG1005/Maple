"""
Move embedding vector creation system for Pokemon moves.
Based on the design document: docs/AI-design/M7/技情報のembeddingベクトル作成.md
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
import os
import pickle
from pathlib import Path


class MoveEmbeddingGenerator:
    """
    Generates embedding vectors for Pokemon moves based on their attributes.
    
    This class processes move data from CSV files and creates numerical embeddings
    that can be used for machine learning models, particularly for RL agents.
    """
    
    def __init__(self, moves_csv_path: str = "config/moves.csv"):
        """
        Initialize the move embedding generator.
        
        Args:
            moves_csv_path: Path to the moves CSV file
        """
        self.moves_csv_path = moves_csv_path
        self.moves_df = None
        self.scaler = MinMaxScaler()
        self.pca = None
        self.sentence_transformer = None
        self.feature_columns = []
        self.embedding_dim = 256  # Default embedding dimension
        self.desc_embedding_dim = 128  # Dimension for text description embeddings
        
    def load_moves_data(self) -> pd.DataFrame:
        """
        Load moves data from CSV file.
        
        Returns:
            DataFrame with moves data
        """
        if not os.path.exists(self.moves_csv_path):
            raise FileNotFoundError(f"Moves CSV file not found: {self.moves_csv_path}")
            
        self.moves_df = pd.read_csv(self.moves_csv_path)
        print(f"Loaded {len(self.moves_df)} moves from {self.moves_csv_path}")
        return self.moves_df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the moves data: handle missing values, type conversions.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.moves_df is None:
            self.load_moves_data()
            
        df = self.moves_df.copy()
        
        # Handle missing values and type conversions
        df['power'] = df['power'].fillna(0).astype(float)
        df['accuracy'] = df['accuracy'].fillna(1.0).astype(float)
        df['pp'] = df['pp'].fillna(0).astype(float)
        df['priority'] = df['priority'].fillna(0).astype(int)
        df['crit_stage'] = df['crit_stage'].fillna(0).astype(int)
        df['multi_hit_min'] = df['multi_hit_min'].fillna(1).astype(int)
        df['multi_hit_max'] = df['multi_hit_max'].fillna(1).astype(int)
        df['recoil_ratio'] = df['recoil_ratio'].fillna(0.0).astype(float)
        df['healing_ratio'] = df['healing_ratio'].fillna(0.0).astype(float)
        
        # Boolean flags
        boolean_flags = [
            'ohko_flag', 'contact_flag', 'sound_flag', 'protectable', 
            'substitutable', 'guard_move_flag', 'recharge_turn', 
            'contenus_effect', 'contenus_turn', 'charging_turn'
        ]
        
        for flag in boolean_flags:
            if flag in df.columns:
                df[flag] = df[flag].fillna(0).astype(float)
        
        # Fill missing descriptions
        df['base_desc'] = df['base_desc'].fillna('通常攻撃。')
        
        print(f"Preprocessed data shape: {df.shape}")
        return df
    
    def create_categorical_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create one-hot encodings for categorical variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with one-hot encoded categorical variables
        """
        # One-hot encode move types
        type_ohe = pd.get_dummies(df['type'], prefix='type')
        
        # One-hot encode move categories
        category_ohe = pd.get_dummies(df['category'], prefix='category')
        
        # Combine with original data
        df_encoded = pd.concat([df, type_ohe, category_ohe], axis=1)
        
        print(f"Added {len(type_ohe.columns)} type features and {len(category_ohe.columns)} category features")
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features to [0,1] range.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled numerical features
        """
        numerical_cols = ['power', 'accuracy', 'pp', 'priority', 'crit_stage',
                         'multi_hit_min', 'multi_hit_max', 'recoil_ratio', 'healing_ratio']
        
        for col in numerical_cols:
            if col in df.columns:
                # Use simple max scaling for better interpretability
                max_val = df[col].max()
                if max_val > 0:
                    df[col + '_scaled'] = df[col] / max_val
                else:
                    df[col + '_scaled'] = df[col]
        
        print(f"Scaled {len(numerical_cols)} numerical features")
        return df
    
    def create_description_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create embeddings for move descriptions using sentence transformers.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with description embeddings
        """
        # Initialize sentence transformer if not already done
        if self.sentence_transformer is None:
            print("Loading sentence transformer model...")
            self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Get descriptions
        descriptions = df['base_desc'].fillna('').tolist()
        
        # Create embeddings
        print(f"Creating embeddings for {len(descriptions)} descriptions...")
        desc_embeddings = self.sentence_transformer.encode(descriptions, normalize_embeddings=True)
        
        # Apply PCA for dimensionality reduction
        # Ensure n_components doesn't exceed min(n_samples, n_features)
        max_components = min(len(descriptions), desc_embeddings.shape[1])
        actual_components = min(self.desc_embedding_dim, max_components)
        
        if self.pca is None:
            self.pca = PCA(n_components=actual_components)
            desc_embeddings_reduced = self.pca.fit_transform(desc_embeddings)
        else:
            desc_embeddings_reduced = self.pca.transform(desc_embeddings)
        
        # Create DataFrame for embeddings
        desc_df = pd.DataFrame(
            desc_embeddings_reduced,
            columns=[f'desc_emb_{i}' for i in range(actual_components)]
        )
        
        # Combine with original data
        df_with_embeddings = pd.concat([df, desc_df], axis=1)
        
        print(f"Added {actual_components} description embedding features")
        return df_with_embeddings
    
    def assemble_final_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Assemble the final feature vector for each move.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Identify feature columns
        type_cols = [col for col in df.columns if col.startswith('type_')]
        category_cols = [col for col in df.columns if col.startswith('category_')]
        scaled_cols = [col for col in df.columns if col.endswith('_scaled')]
        desc_cols = [col for col in df.columns if col.startswith('desc_emb_')]
        
        # Boolean flags
        boolean_flags = [
            'ohko_flag', 'contact_flag', 'sound_flag', 'protectable', 
            'substitutable', 'guard_move_flag', 'recharge_turn', 
            'contenus_effect', 'contenus_turn', 'charging_turn'
        ]
        available_flags = [flag for flag in boolean_flags if flag in df.columns]
        
        # Combine all feature columns
        self.feature_columns = type_cols + category_cols + scaled_cols + available_flags + desc_cols
        
        # Extract feature matrix
        feature_matrix = df[self.feature_columns].values.astype(np.float32)
        
        print(f"Final feature matrix shape: {feature_matrix.shape}")
        print(f"Feature categories: {len(type_cols)} types, {len(category_cols)} categories, "
              f"{len(scaled_cols)} scaled, {len(available_flags)} flags, {len(desc_cols)} descriptions")
        
        return feature_matrix, self.feature_columns
    
    def create_move_embedding_dict(self, df: pd.DataFrame, feature_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create a dictionary mapping move names to their embedding vectors.
        
        Args:
            df: Original DataFrame with move information
            feature_matrix: Feature matrix from assemble_final_features
            
        Returns:
            Dictionary mapping move names to embedding vectors
        """
        move_embeddings = {}
        
        for idx, row in df.iterrows():
            move_name = row['name']
            move_id = row['move_id']
            embedding = feature_matrix[idx]
            
            # Store by both name and ID for flexibility
            move_embeddings[move_name] = embedding
            move_embeddings[f"move_{move_id}"] = embedding
        
        return move_embeddings
    
    def generate_embeddings(self, save_path: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Generate complete move embeddings.
        
        Args:
            save_path: Optional path to save the embeddings
            
        Returns:
            Tuple of (move_embeddings_dict, feature_names)
        """
        print("Starting move embedding generation...")
        
        # Load and preprocess data
        df = self.preprocess_data()
        
        # Create categorical encodings
        df = self.create_categorical_encodings(df)
        
        # Scale numerical features
        df = self.scale_numerical_features(df)
        
        # Create description embeddings
        df = self.create_description_embeddings(df)
        
        # Assemble final features
        feature_matrix, feature_names = self.assemble_final_features(df)
        
        # Create move embedding dictionary
        move_embeddings = self.create_move_embedding_dict(df, feature_matrix)
        
        # Save if requested
        if save_path:
            self.save_embeddings(move_embeddings, feature_names, save_path)
        
        print(f"Generated embeddings for {len(move_embeddings)} moves")
        return move_embeddings, feature_names
    
    def save_embeddings(self, move_embeddings: Dict[str, np.ndarray], 
                       feature_names: List[str], save_path: str):
        """
        Save embeddings to disk.
        
        Args:
            move_embeddings: Dictionary of move embeddings
            feature_names: List of feature names
            save_path: Path to save the embeddings
        """
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        embedding_data = {
            'move_embeddings': move_embeddings,
            'feature_names': feature_names,
            'embedding_dim': len(feature_names),
            'scaler': self.scaler,
            'pca': self.pca
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        print(f"Saved embeddings to {save_path}")
    
    def load_embeddings(self, load_path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Load embeddings from disk.
        
        Args:
            load_path: Path to load the embeddings from
            
        Returns:
            Tuple of (move_embeddings_dict, feature_names)
        """
        with open(load_path, 'rb') as f:
            embedding_data = pickle.load(f)
        
        move_embeddings = embedding_data['move_embeddings']
        feature_names = embedding_data['feature_names']
        
        # Restore components
        self.scaler = embedding_data.get('scaler', self.scaler)
        self.pca = embedding_data.get('pca', self.pca)
        
        print(f"Loaded embeddings for {len(move_embeddings)} moves from {load_path}")
        return move_embeddings, feature_names


def create_move_embeddings(moves_csv_path: str = "config/moves.csv", 
                          save_path: str = "config/move_embeddings.pkl") -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Convenience function to create move embeddings.
    
    Args:
        moves_csv_path: Path to the moves CSV file
        save_path: Path to save the embeddings
        
    Returns:
        Tuple of (move_embeddings_dict, feature_names)
    """
    generator = MoveEmbeddingGenerator(moves_csv_path)
    return generator.generate_embeddings(save_path)


if __name__ == "__main__":
    # Example usage
    move_embeddings, feature_names = create_move_embeddings()
    
    # Print some statistics
    print(f"\nEmbedding Statistics:")
    print(f"Number of moves: {len(move_embeddings)}")
    print(f"Embedding dimension: {len(feature_names)}")
    print(f"Feature names: {feature_names[:10]}...")  # Show first 10 features
    
    # Show example embedding
    example_move = "はたく"
    if example_move in move_embeddings:
        embedding = move_embeddings[example_move]
        print(f"\nExample embedding for '{example_move}':")
        print(f"Shape: {embedding.shape}")
        print(f"First 10 values: {embedding[:10]}")