
import pandas as pd
from src.damage.data_loader import DataLoader
import random

class DamageCalculator:
    def __init__(self, data_loader):
        self.pokemon_data = data_loader.pokemon_stats
        self.pokemon_stats_dict = data_loader.pokemon_stats_dict  # Fast dictionary lookup
        self.move_data = data_loader.moves
        self.type_chart = data_loader.type_chart
        self.move_translations = data_loader.move_translations
        
        # Type conversion mapping from English to Japanese
        self.type_conversion = {
            'Normal': 'ノーマル',
            'Fire': 'ほのお',
            'Water': 'みず',
            'Electric': 'でんき',
            'Grass': 'くさ',
            'Ice': 'こおり',
            'Fighting': 'かくとう',
            'Poison': 'どく',
            'Ground': 'じめん',
            'Flying': 'ひこう',
            'Psychic': 'エスパー',
            'Bug': 'むし',
            'Rock': 'いわ',
            'Ghost': 'ゴースト',
            'Dragon': 'ドラゴン',
            'Dark': 'あく',
            'Steel': 'はがね',
            'Fairy': 'フェアリー'
        }

    def _convert_type_to_japanese(self, type_name):
        """Convert English type name to Japanese for type chart lookup."""
        if type_name in self.type_conversion:
            return self.type_conversion[type_name]
        return type_name  # Return as-is if already Japanese or unknown
    
    def _convert_move_name_to_japanese(self, move_name):
        """Convert English move name to Japanese using move translations."""
        # Try exact match first
        translation = self.move_translations[self.move_translations['English Name'] == move_name]
        if not translation.empty:
            return translation.iloc[0]['Japanese Name']
        
        # Try case-insensitive match and normalize spaces
        move_name_normalized = move_name.replace(' ', '').lower()
        for _, row in self.move_translations.iterrows():
            english_name_normalized = row['English Name'].replace(' ', '').lower()
            if english_name_normalized == move_name_normalized:
                return row['Japanese Name']
        
        return move_name  # Return as-is if not found or already Japanese

    def _calculate_type_effectiveness_from_csv(self, move_type, target_stats):
        """Calculate type effectiveness using CSV data as fallback."""
        defender_type1 = self._convert_type_to_japanese(target_stats['type1'])
        defender_type2 = self._convert_type_to_japanese(target_stats['type2']) if target_stats['type2'] and str(target_stats['type2']) != 'nan' else None
        move_type_japanese = self._convert_type_to_japanese(move_type)
        
        type_effectiveness1_matches = self.type_chart[
            (self.type_chart['attacking_type'] == move_type_japanese) & 
            (self.type_chart['defending_type'] == defender_type1)
        ]
        if type_effectiveness1_matches.empty:
            raise ValueError(f"Type effectiveness not found for {move_type_japanese} ({move_type}) vs {defender_type1}")
        type_effectiveness1 = type_effectiveness1_matches['multiplier'].iloc[0]
        
        type_effectiveness2 = 1.0
        if defender_type2:
            type_effectiveness2_matches = self.type_chart[
                (self.type_chart['attacking_type'] == move_type_japanese) & 
                (self.type_chart['defending_type'] == defender_type2)
            ]
            if not type_effectiveness2_matches.empty:
                type_effectiveness2 = type_effectiveness2_matches['multiplier'].iloc[0]
        
        return type_effectiveness1 * type_effectiveness2

    def _get_modifier(self, modifier_type, attacker, defender, move, field_state):
        modifier = 1.0
        if modifier_type == 'item':
            # TODO: Implement item modifiers
            pass
        elif modifier_type == 'ability':
            # TODO: Implement ability modifiers
            pass
        elif modifier_type == 'weather':
            # TODO: Implement weather modifiers
            pass
        elif modifier_type == 'terrain':
            # TODO: Implement terrain modifiers
            pass
        elif modifier_type == 'guard':
            if defender.get('protect', False):
                modifier *= 0.0 # No damage
            if defender.get('reflect', False) and move.get('category') == 'Physical':
                modifier *= 0.5
            if defender.get('light_screen', False) and move.get('category') == 'Special':
                modifier *= 0.5
            # Roost is more complex, affects type, not direct damage modifier
        elif modifier_type == 'tera':
            # Tera type changes defender's type, already handled by type_effectiveness
            # Tera state might have other effects, but not direct damage modifier for now
            pass
        elif modifier_type == 'status':
            attacker_status = attacker.get('status')
            if attacker_status == 'やけど' and move.get('category') == 'Physical':
                modifier *= 0.5
        return modifier

    def _calculate_max_stat(self, base_stat, level, is_hp):
        # Simplified stat calculation for max EVs (252) and neutral nature (1.0)
        # IVs are assumed to be 31
        if is_hp:
            return int(((base_stat * 2 + 31 + 252/4) * level / 100) + level + 10)
        else:
            return int(((base_stat * 2 + 31 + 252/4) * level / 100) + 5)

    def calculate_damage_range(self, attacker, defender, move, field_state):
        """
        Calculate damage range for a move. Raises exceptions on missing data.
        
        Raises:
            KeyError: If required keys are missing from input dictionaries
            ValueError: If data values are invalid
        """
        if 'attack' not in attacker:
            raise KeyError("Missing 'attack' stat in attacker data")
        if 'defense' not in defender:
            raise KeyError("Missing 'defense' stat in defender data")
        if 'name' not in move:
            raise KeyError("Missing 'name' in move data")
        
        attack_stat = attacker['attack']
        defense_stat = defender['defense']
        
        move_matches = self.move_data[self.move_data['name'] == move['name']]
        if move_matches.empty:
            raise KeyError(f"Move '{move['name']}' not found in move data")
        move_power = move_matches['power'].iloc[0]

        # Rank modifiers
        attack_rank = attacker.get('rank_attack', 0)
        defense_rank = defender.get('rank_defense', 0)
        
        if attack_rank > 0:
            attack_stat *= (2 + attack_rank) / 2
        else:
            attack_stat *= 2 / (2 - attack_rank)

        if defense_rank > 0:
            defense_stat *= (2 + defense_rank) / 2
        else:
            defense_stat *= 2 / (2 - defense_rank)

        # Basic damage calculation
        level = attacker.get('level', 50)  # Level can have a reasonable default
        base_damage = (((2 * level / 5) + 2) * move_power * attack_stat / defense_stat) / 50 + 2

        # Type effectiveness
        move_type = move_matches['type'].iloc[0]
        
        if 'name' not in defender:
            raise KeyError("Missing 'name' in defender data")
        defender_matches = self.pokemon_data[self.pokemon_data['name'] == defender['name']]
        if defender_matches.empty:
            raise KeyError(f"Defender Pokemon '{defender['name']}' not found in pokemon data")
            
        defender_type1 = defender_matches['type1'].iloc[0]
        defender_type2 = defender_matches['type2'].iloc[0]
        
        type_eff1_matches = self.type_chart[(self.type_chart['attacking_type'] == move_type) & (self.type_chart['defending_type'] == defender_type1)]
        if type_eff1_matches.empty:
            raise ValueError(f"Type effectiveness not found for {move_type} vs {defender_type1}")
        type_effectiveness1 = type_eff1_matches['multiplier'].iloc[0]
        
        type_effectiveness2 = 1
        if pd.notna(defender_type2):
            type_eff2_matches = self.type_chart[(self.type_chart['attacking_type'] == move_type) & (self.type_chart['defending_type'] == defender_type2)]
            if type_eff2_matches.empty:
                raise ValueError(f"Type effectiveness not found for {move_type} vs {defender_type2}")
            type_effectiveness2 = type_eff2_matches['multiplier'].iloc[0]
        
        type_effectiveness = type_effectiveness1 * type_effectiveness2
        base_damage *= type_effectiveness

        # STAB (Same Type Attack Bonus)
        if 'name' not in attacker:
            raise KeyError("Missing 'name' in attacker data")
        attacker_matches = self.pokemon_data[self.pokemon_data['name'] == attacker['name']]
        if attacker_matches.empty:
            raise KeyError(f"Attacker Pokemon '{attacker['name']}' not found in pokemon data")
            
        attacker_type1 = attacker_matches['type1'].iloc[0]
        attacker_type2 = attacker_matches['type2'].iloc[0]
        if move_type == attacker_type1 or move_type == attacker_type2:
            base_damage *= 1.5

        # Critical Hit
        critical_hit = False # default
        # Simplified critical hit logic, will be expanded
        if move.get('critical_hit', False):
            critical_hit = True
            base_damage *= 1.5

        item_modifier = self._get_modifier('item', attacker, defender, move, field_state)
        ability_modifier = self._get_modifier('ability', attacker, defender, move, field_state)
        weather_modifier = self._get_modifier('weather', attacker, defender, move, field_state)
        terrain_modifier = self._get_modifier('terrain', attacker, defender, move, field_state)
        guard_modifier = self._get_modifier('guard', attacker, defender, move, field_state)
        tera_modifier = self._get_modifier('tera', attacker, defender, move, field_state)
        status_modifier = self._get_modifier('status', attacker, defender, move, field_state)

        base_damage *= (item_modifier * ability_modifier * weather_modifier * terrain_modifier * guard_modifier * tera_modifier * status_modifier)

        # Damage range (0.85 to 1.0)
        min_damage = int(base_damage * 0.85)
        max_damage = int(base_damage * 1.0)

        # Calculate HP percentage and knockout count
        defender_max_hp = defender.get('max_hp') # Assuming max_hp is available in defender object
        hp_percentage_min = (min_damage / defender_max_hp) * 100 if defender_max_hp else 0
        hp_percentage_max = (max_damage / defender_max_hp) * 100 if defender_max_hp else 0

        knockout_count_text = ""
        if defender_max_hp:
            if min_damage >= defender_max_hp:
                knockout_count_text = "確定1発"
            elif max_damage < defender_max_hp:
                # Calculate probability of 2HKO, 3HKO etc.
                # This is a simplified approach for now
                if min_damage * 2 >= defender_max_hp:
                    knockout_count_text = "乱数2発"
                elif min_damage * 3 >= defender_max_hp:
                    knockout_count_text = "乱数3発"
                else:
                    knockout_count_text = "確定複数発"
            else:
                knockout_count_text = "乱数1発" # max_damage can KO, but min_damage cannot

        # Calculation details
        calculation_details = {
            'type_effectiveness': type_effectiveness,
            'stab': 1.5 if (move_type == attacker_type1 or move_type == attacker_type2) else 1.0,
            'critical_hit': critical_hit,
            'attack_rank_modifier': (2 + attack_rank) / 2 if attack_rank > 0 else 2 / (2 - attack_rank),
            'defense_rank_modifier': (2 + defense_rank) / 2 if defense_rank > 0 else 2 / (2 - defense_rank),
            # TODO: Add item, ability, weather, terrain, etc. details
        }

        return {
            'damage_range': (min_damage, max_damage),
            'hp_percentage': (hp_percentage_min, hp_percentage_max),
            'knockout_count': knockout_count_text,
            'calculation_details': calculation_details
        }

    def simulate_move_effect(self, attacker, defender, move, field_state):
        import random

        # 命中判定
        accuracy = move.get('accuracy', 100) # 技の命中率
        # TODO: 場の状態、双方の命中・回避ランクを考慮
        if random.randint(1, 100) > accuracy:
            return {'hit': False, 'damage': 0, 'critical_hit': False, 'additional_effect': {'triggered': False, 'content': None}}

        # ダメージ計算
        # calculate_damage_range と同様のロジックを使用し、乱数を単一の値に固定
        # Simplified example, needs to be fleshed out with actual data objects
        attack_stat = attacker.get('attack')
        defense_stat = defender.get('defense')
        move_power = self.move_data[self.move_data['name'] == move['name']]['power'].iloc[0]

        base_damage = (((2 * attacker.get('level', 50) / 5) + 2) * move_power * attack_stat / defense_stat) / 50 + 2

        # Type effectiveness
        move_type = self.move_data[self.move_data['name'] == move['name']]['type'].iloc[0]
        defender_type1 = self.pokemon_data[self.pokemon_data['name'] == defender['name']]['type1'].iloc[0]
        defender_type2 = self.pokemon_data[self.pokemon_data['name'] == defender['name']]['type2'].iloc[0]
        
        type_effectiveness1 = self.type_chart[(self.type_chart['attacking_type'] == move_type) & (self.type_chart['defending_type'] == defender_type1)]['multiplier'].iloc[0]
        type_effectiveness2 = 1
        if pd.notna(defender_type2):
            type_effectiveness2 = self.type_chart[(self.type_chart['attacking_type'] == move_type) & (self.type_chart['defending_type'] == defender_type2)]['multiplier'].iloc[0]
        
        type_effectiveness = type_effectiveness1 * type_effectiveness2
        base_damage *= type_effectiveness

        # STAB (Same Type Attack Bonus)
        attacker_type1 = self.pokemon_data[self.pokemon_data['name'] == attacker['name']]['type1'].iloc[0]
        attacker_type2 = self.pokemon_data[self.pokemon_data['name'] == attacker['name']]['type2'].iloc[0]
        if move_type == attacker_type1 or move_type == attacker_type2:
            base_damage *= 1.5

        # Critical Hit (randomly determined for simulation)
        critical_hit = random.random() < 0.0625 # 1/16 chance for critical hit
        if critical_hit:
            base_damage *= 1.5

        # Random damage multiplier (0.85 to 1.0)
        final_damage = int(base_damage * random.uniform(0.85, 1.0))

        # 追加効果判定
        additional_effect_triggered = False
        additional_effect_content = None
        effect_prob = move.get('effect_prob', 0)
        if random.randint(1, 100) <= effect_prob:
            additional_effect_triggered = True
            additional_effect_content = move.get('effect_type', 'Unknown effect') # Placeholder

        return {
            'hit': True,
            'damage': final_damage,
            'critical_hit': critical_hit,
            'additional_effect': {
                'triggered': additional_effect_triggered,
                'content': additional_effect_content
            }
        }

    def calculate_damage_expectation_for_ai(self, attacker_stats, target_pokemon, move_object, move_type):
        """
        Calculate damage expectation for AI state space observation.
        
        Args:
            attacker_stats (dict): Dictionary containing attacker information
                - 'attack' or 'special_attack': real stat value
                - 'rank_attack' or 'rank_special_attack': rank change (-6 to +6)
                - 'type1', 'type2': attacker types
                - 'tera_type': tera type (optional)
                - 'is_terastalized': boolean indicating tera status
                - 'level': pokemon level (default 50)
            target_pokemon: Pokemon object with base_stats and type information
            move_object: Move object with base_power and category information
            move_type (str): Type of the move (English or Japanese)
            
        Returns:
            tuple: (expected_damage_percent, variance_percent)
                   e.g., (46.5, 1.3) for 45.2~47.8% damage
                   
        Raises:
            KeyError: If target Pokemon or move not found in data
            ValueError: If required data is missing or invalid
            Exception: For any calculation errors
        """
        # Extract names for error messages
        target_name = target_pokemon.species if hasattr(target_pokemon, 'species') else str(target_pokemon)
        move_name = move_object.id if hasattr(move_object, 'id') else str(move_object)
        
        # Get target base stats from Pokemon object
        if hasattr(target_pokemon, 'base_stats') and target_pokemon.base_stats:
            pokemon_base_stats = target_pokemon.base_stats
            
            # Convert poke-env base_stats format to expected format
            target_stats = {
                'HP': pokemon_base_stats.get('hp', 50),
                'atk': pokemon_base_stats.get('atk', 50), 
                'def': pokemon_base_stats.get('def', 50),
                'spa': pokemon_base_stats.get('spa', 50),
                'spd': pokemon_base_stats.get('spd', 50),
                'spe': pokemon_base_stats.get('spe', 50),
                'type1': target_pokemon.type_1.name if hasattr(target_pokemon, 'type_1') and target_pokemon.type_1 else 'Normal',
                'type2': target_pokemon.type_2.name if hasattr(target_pokemon, 'type_2') and target_pokemon.type_2 else None
            }
        else:
            # Fallback to CSV lookup
            if target_name not in self.pokemon_stats_dict:
                raise KeyError(f"Target Pokemon '{target_name}' not found in pokemon_stats data and no base_stats available")
            target_stats = self.pokemon_stats_dict[target_name]
        
        # Get move power and category from Move object
        if hasattr(move_object, 'base_power') and hasattr(move_object, 'category'):
            move_power = move_object.base_power
            move_category_enum = move_object.category
            
            # Convert MoveCategory enum to Japanese string
            if hasattr(move_category_enum, 'name'):
                if move_category_enum.name.lower() == 'physical':
                    move_category = '物理'
                elif move_category_enum.name.lower() == 'special':
                    move_category = '特殊'
                else:
                    move_category = '変化'  # Status moves
            else:
                move_category = '物理'  # Default fallback
                
        else:
            # Fallback to CSV method
            japanese_move_name = self._convert_move_name_to_japanese(move_name)
            move_data = self.move_data[self.move_data['name'] == japanese_move_name]
            if move_data.empty:
                raise KeyError(f"Move '{japanese_move_name}' (from '{move_name}') not found in moves data")
            
            move_power = move_data.iloc[0]['base_power']
            move_category = move_data.iloc[0]['category']
        
        # Skip non-damaging moves (変化技)
        if move_category == '変化' or pd.isna(move_power) or move_power <= 0:
            return (0.0, 0.0)
        
        # Determine attack and defense stats based on move category
        if move_category == '物理':
            if 'attack' not in attacker_stats:
                raise ValueError(f"Physical attack stat required for move '{move_name}' but not provided")
            attack_stat = attacker_stats['attack']
            attack_rank = attacker_stats.get('rank_attack', 0)
            defense_stat = self._calculate_max_stat(target_stats['def'], attacker_stats.get('level', 50), False)
        elif move_category == '特殊':
            if 'special_attack' not in attacker_stats:
                raise ValueError(f"Special attack stat required for move '{move_name}' but not provided")
            attack_stat = attacker_stats['special_attack']
            attack_rank = attacker_stats.get('rank_special_attack', 0)
            defense_stat = self._calculate_max_stat(target_stats['spd'], attacker_stats.get('level', 50), False)
        else:
            raise ValueError(f"Unknown move category '{move_category}' for move '{move_name}'")
        
        # Apply rank modifiers
        if attack_rank > 0:
            attack_stat *= (2 + attack_rank) / 2
        elif attack_rank < 0:
            attack_stat *= 2 / (2 - attack_rank)
        
        # Basic damage calculation
        level = attacker_stats.get('level', 50)
        base_damage = (((2 * level / 5) + 2) * move_power * attack_stat / defense_stat) / 50 + 2
        
        # Type effectiveness using poke-env's damage_multiplier method
        if hasattr(target_pokemon, 'damage_multiplier'):
            # Use Pokemon object's built-in damage calculation
            # Need to create a PokemonType from move_type string
            from poke_env.environment.pokemon_type import PokemonType
            try:
                move_pokemon_type = PokemonType.from_name(move_type.lower())
                type_effectiveness = target_pokemon.damage_multiplier(move_pokemon_type)
            except Exception as e:
                # Fallback to CSV method
                type_effectiveness = self._calculate_type_effectiveness_from_csv(move_type, target_stats)
        else:
            # Fallback to CSV method
            type_effectiveness = self._calculate_type_effectiveness_from_csv(move_type, target_stats)
        base_damage *= type_effectiveness
        
        # STAB (Same Type Attack Bonus)
        attacker_type1 = self._convert_type_to_japanese(attacker_stats.get('type1', ''))
        attacker_type2 = self._convert_type_to_japanese(attacker_stats.get('type2', ''))
        tera_type = self._convert_type_to_japanese(attacker_stats.get('tera_type', ''))
        is_terastalized = attacker_stats.get('is_terastalized', False)
        
        # Apply STAB based on tera status
        if is_terastalized and tera_type:
            if move_type == tera_type:
                base_damage *= 1.5  # Tera STAB
        else:
            if move_type == attacker_type1 or move_type == attacker_type2:
                base_damage *= 1.5  # Regular STAB
        
        # Calculate target max HP
        target_max_hp = self._calculate_max_stat(target_stats['HP'], level, True)
        
        # Calculate damage range (0.85 to 1.0)
        min_damage = base_damage * 0.85
        max_damage = base_damage * 1.0
        
        # Convert to percentage
        min_percent = (min_damage / target_max_hp) * 100
        max_percent = (max_damage / target_max_hp) * 100
        
        # Calculate expected value and variance
        expected_percent = (min_percent + max_percent) / 2
        variance_percent = (max_percent - min_percent) / 2
        
        
        return (expected_percent, variance_percent)
