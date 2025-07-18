"""
Move embedding vector creation system for Pokemon moves.
Based on the design documents: 
- docs/AI-design/M7/技情報のembeddingベクトル作成.md
- docs/AI-design/M7/技の説明をベクトル化.md
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
import re
from pathlib import Path


class MoveEmbeddingGenerator:
    """
    Generates embedding vectors for Pokemon moves based on their attributes.
    
    This class processes move data from CSV files and creates numerical embeddings
    that can be used for machine learning models, particularly for RL agents.
    """
    
    def __init__(self, moves_csv_path: str = "config/moves.csv", 
                 japanese_model: bool = True):
        """
        Initialize the move embedding generator.
        
        Args:
            moves_csv_path: Path to the moves CSV file
            japanese_model: Whether to use Japanese-specific model for descriptions
        """
        self.moves_csv_path = moves_csv_path
        self.moves_df = None
        self.scaler = MinMaxScaler()
        self.pca = None
        self.sentence_transformer = None
        self.feature_columns = []
        self.embedding_dim = 256  # Default embedding dimension
        self.desc_embedding_dim = 128  # Dimension for text description embeddings
        self.japanese_model = japanese_model
        self.use_advanced_preprocessing = True  # Enhanced Japanese text processing
        
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
    
    def preprocess_japanese_text(self, text: str) -> str:
        """
        Preprocess Japanese text for better embedding quality.
        Based on the design document: docs/AI-design/M7/技の説明をベクトル化.md
        
        Args:
            text: Input Japanese text
            
        Returns:
            Preprocessed text
        """
        if not text or pd.isna(text):
            return "通常攻撃。"
            
        # Remove special characters and normalize
        text = str(text).strip()
        
        # Remove excessive whitespace and line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        
        # Remove special symbols that don't add semantic meaning
        text = re.sub(r'[【】〈〉《》「」『』]', '', text)
        
        # Normalize katakana/hiragana inconsistencies (optional)
        # Keep original for now as Pokemon terms are often in katakana
        
        # Truncate to 256 characters to avoid token limits
        if len(text) > 256:
            text = text[:256]
            
        # Ensure text ends properly
        if text and not text.endswith('。'):
            text += '。'
            
        return text
    
    def get_sentence_transformer_model(self) -> str:
        """
        Get the appropriate sentence transformer model based on configuration.
        Based on the design document recommendations.
        
        Returns:
            Model name for sentence transformers
        """
        if self.japanese_model:
            # Japanese-specific models (recommended in the document)
            # Try multiple models in order of preference
            japanese_models = [
                'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',  # Best multilingual
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',  # Good balance
                'sentence-transformers/paraphrase-MiniLM-L6-v2'  # Fallback
            ]
            
            # For now, use the multilingual model that works well with Japanese
            return japanese_models[0]
        else:
            # Original multilingual model
            return 'paraphrase-multilingual-MiniLM-L12-v2'
    
    def create_description_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create embeddings for move descriptions using sentence transformers.
        Enhanced with Japanese text preprocessing and better model selection.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with description embeddings
        """
        # Initialize sentence transformer if not already done
        if self.sentence_transformer is None:
            model_name = self.get_sentence_transformer_model()
            print(f"Loading sentence transformer model: {model_name}")
            try:
                self.sentence_transformer = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Failed to load {model_name}, falling back to default multilingual model")
                self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Get descriptions with enhanced preprocessing
        raw_descriptions = df['base_desc'].fillna('').tolist()
        
        if self.use_advanced_preprocessing:
            print("Applying advanced Japanese text preprocessing...")
            descriptions = [self.preprocess_japanese_text(desc) for desc in raw_descriptions]
        else:
            descriptions = [str(desc) if desc else "通常攻撃。" for desc in raw_descriptions]
        
        # Create embeddings
        print(f"Creating embeddings for {len(descriptions)} descriptions...")
        desc_embeddings = self.sentence_transformer.encode(
            descriptions, 
            normalize_embeddings=True,
            batch_size=32,  # Process in batches for better performance
            show_progress_bar=False
        )
        
        # Apply PCA for dimensionality reduction
        # Ensure n_components doesn't exceed min(n_samples, n_features)
        max_components = min(len(descriptions), desc_embeddings.shape[1])
        actual_components = min(self.desc_embedding_dim, max_components)
        
        if self.pca is None:
            self.pca = PCA(n_components=actual_components)
            desc_embeddings_reduced = self.pca.fit_transform(desc_embeddings)
            print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_[:5].sum():.3f} (first 5 components)")
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
        print(f"Original embedding dimension: {desc_embeddings.shape[1]}")
        print(f"Reduced embedding dimension: {actual_components}")
        
        return df_with_embeddings
    
    def assemble_final_features(self, df: pd.DataFrame, 
                               fusion_strategy: str = "concatenate") -> Tuple[np.ndarray, List[str]]:
        """
        Assemble the final feature vector for each move with enhanced fusion strategy.
        Based on the design document: docs/AI-design/M7/技の説明をベクトル化.md
        
        Args:
            df: Processed DataFrame
            fusion_strategy: Strategy for combining features ('concatenate', 'balanced', 'weighted')
            
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
        
        # Structured features (non-text)
        structured_cols = type_cols + category_cols + scaled_cols + available_flags
        
        if fusion_strategy == "concatenate":
            # Simple concatenation as before
            self.feature_columns = structured_cols + desc_cols
            feature_matrix = df[self.feature_columns].values.astype(np.float32)
            
        elif fusion_strategy == "balanced":
            # Balance structured and text features to equal dimensions
            # This follows the document's recommendation: 128 + 128 → 256
            structured_matrix = df[structured_cols].values.astype(np.float32)
            text_matrix = df[desc_cols].values.astype(np.float32)
            
            # If text features are more than structured, they're already reduced by PCA
            # If structured features are more, we could apply PCA here too
            target_dim = min(len(desc_cols), 128)  # Target balanced dimension
            
            if structured_matrix.shape[1] > target_dim:
                # Apply PCA to structured features too
                from sklearn.decomposition import PCA
                struct_pca = PCA(n_components=target_dim)
                structured_matrix = struct_pca.fit_transform(structured_matrix)
                struct_cols_names = [f'struct_{i}' for i in range(target_dim)]
            else:
                struct_cols_names = structured_cols
                
            # Combine balanced features
            feature_matrix = np.concatenate([structured_matrix, text_matrix], axis=1)
            self.feature_columns = struct_cols_names + desc_cols
            
        elif fusion_strategy == "weighted":
            # Weighted combination giving more importance to text features
            # Based on the document's emphasis on semantic understanding
            structured_matrix = df[structured_cols].values.astype(np.float32)
            text_matrix = df[desc_cols].values.astype(np.float32)
            
            # Apply weights: 0.3 for structured, 0.7 for text (semantic)
            structured_weight = 0.3
            text_weight = 0.7
            
            # Normalize matrices first
            structured_matrix = (structured_matrix - structured_matrix.mean(axis=0)) / (structured_matrix.std(axis=0) + 1e-8)
            text_matrix = (text_matrix - text_matrix.mean(axis=0)) / (text_matrix.std(axis=0) + 1e-8)
            
            # Apply weights
            structured_matrix *= structured_weight
            text_matrix *= text_weight
            
            # Combine weighted features
            feature_matrix = np.concatenate([structured_matrix, text_matrix], axis=1)
            self.feature_columns = [f'struct_w_{col}' for col in structured_cols] + [f'text_w_{col}' for col in desc_cols]
            
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        print(f"Final feature matrix shape: {feature_matrix.shape}")
        print(f"Fusion strategy: {fusion_strategy}")
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
    
    def generate_embeddings(self, save_path: Optional[str] = None, 
                           fusion_strategy: str = "concatenate") -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Generate complete move embeddings with enhanced fusion strategies.
        
        Args:
            save_path: Optional path to save the embeddings
            fusion_strategy: Strategy for combining features ('concatenate', 'balanced', 'weighted')
            
        Returns:
            Tuple of (move_embeddings_dict, feature_names)
        """
        print("Starting enhanced move embedding generation...")
        print(f"Fusion strategy: {fusion_strategy}")
        print(f"Japanese model: {self.japanese_model}")
        print(f"Advanced preprocessing: {self.use_advanced_preprocessing}")
        
        # Load and preprocess data
        df = self.preprocess_data()
        
        # Create categorical encodings
        df = self.create_categorical_encodings(df)
        
        # Scale numerical features
        df = self.scale_numerical_features(df)
        
        # Create description embeddings with enhanced processing
        df = self.create_description_embeddings(df)
        
        # Assemble final features with fusion strategy
        feature_matrix, feature_names = self.assemble_final_features(df, fusion_strategy)
        
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
            'feature_columns': self.feature_columns,  # Store feature columns for semantic search
            'embedding_dim': len(feature_names),
            'scaler': self.scaler,
            'pca': self.pca,
            'japanese_model': self.japanese_model,
            'desc_embedding_dim': self.desc_embedding_dim
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
        self.feature_columns = embedding_data.get('feature_columns', self.feature_columns)
        self.japanese_model = embedding_data.get('japanese_model', self.japanese_model)
        self.desc_embedding_dim = embedding_data.get('desc_embedding_dim', self.desc_embedding_dim)
        
        print(f"Loaded embeddings for {len(move_embeddings)} moves from {load_path}")
        print(f"Feature columns restored: {len(self.feature_columns) if self.feature_columns else 0}")
        return move_embeddings, feature_names
    
    def semantic_search(self, query: str, move_embeddings: Dict[str, np.ndarray], 
                       top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform semantic search for moves based on natural language query.
        Based on the design document: docs/AI-design/M7/技の説明をベクトル化.md
        
        Args:
            query: Natural language query (e.g., "リフレクター系", "味方強化")
            move_embeddings: Dictionary of move embeddings
            top_k: Number of top results to return
            
        Returns:
            List of tuples (move_name, similarity_score)
        """
        if self.sentence_transformer is None:
            model_name = self.get_sentence_transformer_model()
            self.sentence_transformer = SentenceTransformer(model_name)
        
        # Preprocess query
        processed_query = self.preprocess_japanese_text(query)
        
        # Encode query
        query_embedding = self.sentence_transformer.encode(
            [processed_query], 
            normalize_embeddings=True
        )[0]
        
        # Apply same PCA transformation if available
        if self.pca is not None:
            query_embedding = self.pca.transform(query_embedding.reshape(1, -1))[0]
        
        # Calculate similarities
        similarities = []
        for move_name, move_embedding in move_embeddings.items():
            if move_name.startswith('move_'):
                continue  # Skip move_id entries
            
            # For semantic search, we want to use the description embedding portion
            # Find the description embedding columns
            desc_cols = [col for col in self.feature_columns if col.startswith('desc_emb_')]
            if desc_cols:
                # Get indices of description embedding columns
                desc_indices = [i for i, col in enumerate(self.feature_columns) 
                              if col.startswith('desc_emb_')]
                
                if desc_indices:
                    # Extract description embedding portion
                    move_desc_embedding = move_embedding[desc_indices]
                    
                    # Ensure both embeddings have the same length
                    if len(query_embedding) == len(move_desc_embedding):
                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding, move_desc_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(move_desc_embedding) + 1e-8
                        )
                        similarities.append((move_name, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_move_embeddings(moves_csv_path: str = "config/moves.csv", 
                          save_path: str = "config/move_embeddings.pkl",
                          fusion_strategy: str = "concatenate",
                          japanese_model: bool = True) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Convenience function to create move embeddings with enhanced options.
    
    Args:
        moves_csv_path: Path to the moves CSV file
        save_path: Path to save the embeddings
        fusion_strategy: Strategy for combining features ('concatenate', 'balanced', 'weighted')
        japanese_model: Whether to use Japanese-specific model
        
    Returns:
        Tuple of (move_embeddings_dict, feature_names)
    """
    generator = MoveEmbeddingGenerator(moves_csv_path, japanese_model=japanese_model)
    return generator.generate_embeddings(save_path, fusion_strategy=fusion_strategy)


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