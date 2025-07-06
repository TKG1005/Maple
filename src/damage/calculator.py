from .data_loader import DataLoader
import math

class DamageCalculator:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def calculate_damage_range(self, attacker: dict, defender: dict, move_name: str, field_state: dict = None):
        """
        Calculates the damage range for a given move.

        Args:
            attacker (dict): Attacking pokemon's state.
                Example: {'name': 'Pikachu', 'level': 50, 'stats': {'attack': 100, 'sp_attack': 120}}
            defender (dict): Defending pokemon's state.
                Example: {'name': 'Snorlax', 'stats': {'defense': 150, 'sp_defense': 180, 'hp': 200}}
            move_name (str): The name of the move.
            field_state (dict): The state of the field (weather, etc.). Ignored for now.

        Returns:
            dict: A dictionary containing the damage range and other info.
        """
        move_data_list = self.data_loader.get_move_data(move_name)
        if not move_data_list:
            raise ValueError(f"Move '{move_name}' not found in data.")
        move_data = move_data_list[0]

        # 1. Determine Attack and Defense stats
        if move_data['category'] == '物理':
            attack_stat = attacker['stats']['attack']
            defense_stat = defender['stats']['defense']
        elif move_data['category'] == '特殊':
            attack_stat = attacker['stats']['sp_attack']
            defense_stat = defender['stats']['sp_defense']
        else: # For non-damaging moves
            return {'min_damage': 0, 'max_damage': 0, 'min_hp_percentage': 0, 'max_hp_percentage': 0, 'ko_chance': 'No damage'}

        # 2. Basic damage formula
        # Damage = ( (level * 2 / 5 + 2) * power * A / D / 50 + 2 )
        level = attacker['level']
        power = move_data['power']
        
        base_damage = math.floor(math.floor((level * 2 / 5) + 2) * power * attack_stat / defense_stat / 50) + 2

        # 3. Calculate min/max damage based on random multiplier (0.85 to 1.0)
        min_damage = math.floor(base_damage * 0.85)
        max_damage = base_damage

        # 4. Calculate HP percentage and KO chance (simplified)
        defender_hp = defender['stats']['hp']
        if defender_hp <= 0:
            return {'min_damage': min_damage, 'max_damage': max_damage, 'min_hp_percentage': float('inf'), 'max_hp_percentage': float('inf'), 'ko_chance': 'Already fainted'}

        min_hp_percentage = round((min_damage / defender_hp) * 100, 1)
        max_hp_percentage = round((max_damage / defender_hp) * 100, 1)

        if max_damage == 0: # Avoid division by zero
            ko_chance_str = "No damage"
        elif max_damage >= defender_hp:
            ko_chance_str = "確定1発"
        elif min_damage * 2 >= defender_hp:
            ko_chance_str = "確定2発"
        elif max_damage * 2 >= defender_hp:
            ko_chance_str = "乱数2発"
        else:
            ko_chance_str = f"確定{math.ceil(defender_hp / max_damage)}発"

        return {
            'min_damage': min_damage,
            'max_damage': max_damage,
            'min_hp_percentage': min_hp_percentage,
            'max_hp_percentage': max_hp_percentage,
            'ko_chance': ko_chance_str
        }

    def simulate_move_effect(self, attacker, defender, move, field_state):
        # TODO: Implement move simulation logic
        pass
