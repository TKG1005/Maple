import pandas as pd
import os

class DataLoader:

    def __init__(self, data_dir='config'):
        self.data_dir = data_dir
        self.pokemon_stats = self.load_pokemon_stats(os.path.join(self.data_dir, 'pokemon_stats.csv'))
        self.moves = self.load_moves(os.path.join(self.data_dir, 'pokemon_all_moves.csv'))
        self.type_chart = self.load_type_chart(os.path.join(self.data_dir, 'type_chart.csv'))
        self.move_translations = self.load_move_translations(os.path.join(self.data_dir, 'moves_english_japanese.csv'))

    def load_pokemon_stats(self, file_path):
        df = pd.read_csv(file_path, on_bad_lines='skip')
        # Create dictionary index for faster lookups
        self.pokemon_stats_dict = df.set_index('name').to_dict('index')
        return df

    def load_moves(self, file_path):
        df = pd.read_csv(file_path)
        # Convert numeric columns to proper types
        df['base_power'] = pd.to_numeric(df['base_power'], errors='coerce')
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        df['PP'] = pd.to_numeric(df['PP'], errors='coerce')
        return df

    def load_type_chart(self, file_path):
        return pd.read_csv(file_path)
    
    def load_move_translations(self, file_path):
        # Handle CSV with commas in names - need to manually parse the problematic rows
        import csv
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) > 2:
                    # Handle cases where English name has commas
                    english_name = ','.join(row[:-1])  # Join all but last column
                    japanese_name = row[-1]
                    rows.append([english_name, japanese_name])
                else:
                    rows.append(row)
        
        return pd.DataFrame(rows, columns=['English Name', 'Japanese Name'])
