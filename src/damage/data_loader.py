import pandas as pd
import os

class DataLoader:

    def __init__(self, data_dir='config'):
        self.data_dir = data_dir
        self.pokemon_stats = self.load_pokemon_stats(os.path.join(self.data_dir, 'pokemon_stats.csv'))
        self.moves = self.load_moves(os.path.join(self.data_dir, 'moves.csv'))
        self.type_chart = self.load_type_chart(os.path.join(self.data_dir, 'type_chart.csv'))

    def load_pokemon_stats(self, file_path):
        return pd.read_csv(file_path)

    def load_moves(self, file_path):
        return pd.read_csv(file_path)

    def load_type_chart(self, file_path):
        return pd.read_csv(file_path)
