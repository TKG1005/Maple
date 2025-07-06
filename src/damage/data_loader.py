import pandas as pd
import os

class DataLoader:
    def __init__(self, config_path='config'):
        self.pokemon_stats = self._load_csv(os.path.join(config_path, 'pokemon_stats.csv'))
        self.moves = self._load_csv(os.path.join(config_path, 'moves.csv'))
        self.type_chart = self._load_csv(os.path.join(config_path, 'type_chart.csv'))

    def _load_csv(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        return pd.read_csv(file_path)

    def get_pokemon_data(self, name):
        return self.pokemon_stats[self.pokemon_stats['name'] == name].to_dict('records')

    def get_move_data(self, name):
        return self.moves[self.moves['name'] == name].to_dict('records')

    def get_type_multiplier(self, attacking_type, defending_type):
        multiplier = self.type_chart[
            (self.type_chart['attacking_type'] == attacking_type) &
            (self.type_chart['defending_type'] == defending_type)
        ]['multiplier']
        if multiplier.empty:
            return 1.0
        return multiplier.iloc[0]