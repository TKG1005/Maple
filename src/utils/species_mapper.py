"""
Pokemon species name to Pokedex number mapping utility.
Optimized for frequent access during battle state observation.
"""

import pandas as pd
import os
from typing import Dict, Optional


class SpeciesMapper:
    """Efficient Pokemon species name to Pokedex number mapping."""
    
    def __init__(self, csv_path: Optional[str] = None):
        """Initialize the species mapper with caching for performance."""
        if csv_path is None:
            # Default path relative to project root
            csv_path = os.path.join(
                os.path.dirname(__file__), 
                "..", "..", "config", "pokemon_stats.csv"
            )
        
        self._species_to_id: Dict[str, int] = {}
        self._id_to_species: Dict[int, str] = {}
        self._initialized = False
        self._csv_path = csv_path
        
    def _load_mappings(self):
        """Load mappings from CSV file. Called lazily on first access."""
        if self._initialized:
            return
            
        try:
            # Check if CSV exists and has the expected format
            if not os.path.exists(self._csv_path):
                print(f"Warning: Pokemon stats CSV not found at {self._csv_path}")
                self._initialize_fallback()
                return
                
            df = pd.read_csv(self._csv_path, on_bad_lines='skip')
            
            # Handle different possible column names
            name_col = None
            id_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'name' in col_lower and name_col is None:
                    name_col = col
                elif ('no' in col_lower or 'id' in col_lower or 'number' in col_lower) and id_col is None:
                    id_col = col
                    
            if name_col is None or id_col is None:
                print(f"Warning: Could not find name/id columns in {self._csv_path}")
                self._initialize_fallback()
                return
                
            # Build mappings
            for _, row in df.iterrows():
                try:
                    name = str(row[name_col]).lower().strip()
                    pokedex_id = int(row[id_col])
                    
                    self._species_to_id[name] = pokedex_id
                    self._id_to_species[pokedex_id] = name
                    
                except (ValueError, TypeError):
                    continue  # Skip invalid rows
                    
            # Add special entries
            self._species_to_id['unknown'] = 0
            self._species_to_id['none'] = 0
            self._id_to_species[0] = 'unknown'
            
            print(f"Loaded {len(self._species_to_id)} Pokemon species mappings")
            
        except Exception as e:
            print(f"Error loading Pokemon stats CSV: {e}")
            self._initialize_fallback()
            
        self._initialized = True
        
    def _initialize_fallback(self):
        """Initialize with fallback data if CSV loading fails."""
        # Minimal fallback mappings for common Pokemon
        fallback_data = {
            'unknown': 0, 'none': 0,
            'pikachu': 25, 'charizard': 6, 'blastoise': 9, 'venusaur': 3,
            'mewtwo': 150, 'mew': 151, 'dragonite': 149,
        }
        
        self._species_to_id = fallback_data.copy()
        self._id_to_species = {v: k for k, v in fallback_data.items()}
        
        print("Warning: Using fallback Pokemon species mappings")
        
    def get_pokedex_id(self, species_name: str) -> int:
        """Get Pokedex ID for a species name. Returns 0 for unknown species."""
        if not self._initialized:
            self._load_mappings()
            
        if species_name is None:
            return 0
            
        # Normalize the species name for lookup
        normalized_name = str(species_name).lower().strip()
        
        # Handle poke-env specific naming patterns
        if normalized_name.startswith('pokemon_'):
            normalized_name = normalized_name[8:]  # Remove 'pokemon_' prefix
            
        return self._species_to_id.get(normalized_name, 0)
        
    def get_species_name(self, pokedex_id: int) -> str:
        """Get species name for a Pokedex ID. Returns 'unknown' for invalid IDs."""
        if not self._initialized:
            self._load_mappings()
            
        return self._id_to_species.get(pokedex_id, 'unknown')
        
    def get_team_pokedex_ids(self, team_list) -> list:
        """Convert a list of Pokemon to their Pokedex IDs efficiently."""
        if not team_list:
            return [0] * 6  # Return 6 zeros for empty team
            
        result = []
        for pokemon in team_list:
            if pokemon is None:
                result.append(0)
            else:
                # Handle both string names and Pokemon objects
                if hasattr(pokemon, 'species'):
                    species_name = pokemon.species
                else:
                    species_name = str(pokemon)
                result.append(self.get_pokedex_id(species_name))
                
        # Pad with zeros to ensure 6 entries
        while len(result) < 6:
            result.append(0)
            
        return result[:6]  # Ensure exactly 6 entries


# Global instance for efficient reuse
_global_mapper: Optional[SpeciesMapper] = None


def get_species_mapper() -> SpeciesMapper:
    """Get the global species mapper instance."""
    global _global_mapper
    if _global_mapper is None:
        _global_mapper = SpeciesMapper()
    return _global_mapper