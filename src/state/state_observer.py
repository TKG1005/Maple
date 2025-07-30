# src/state/state_observer.py

import numpy as np
import yaml

# import time # timeは現在使われていないようなのでコメントアウトまたは削除してよい
from poke_env.environment.abstract_battle import AbstractBattle

# PokemonType のインポートが不足している可能性があるため追加 (エンコーダーなどで利用する場合)
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.move_category import MoveCategory  # MoveCategoryも追加
from src.state.type_matchup_extractor import TypeMatchupFeatureExtractor
from src.utils.species_mapper import get_species_mapper
from src.damage.calculator import DamageCalculator
from src.damage.data_loader import DataLoader
# Avoid circular import by importing MoveEmbeddingLayer only when needed


class StateObserver:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.spec = yaml.safe_load(f)
        # _build_encoders は get_observation_dimension でも利用する可能性があるため、先に初期化
        self.encoders = self._build_encoders(self.spec)
        self.opp_total_estimate = 3  # 敵の手持ちの初期値
        self.type_matchup_extractor = TypeMatchupFeatureExtractor()
        
        # Initialize species mapper for efficient Pokedex number conversion
        self.species_mapper = get_species_mapper()
        
        # Initialize damage calculator for damage expectation features
        # Use lazy initialization to avoid loading overhead if not needed
        self._damage_calculator = None
        self._damage_calculator_initialized = False
        
        # Initialize move embedding layer for move representation
        # Use lazy initialization to avoid loading overhead if not needed
        self._move_embedding_layer = None
        self._move_embedding_initialized = False
        
        # Initialize species embedding layer for normalized Pokemon species representations
        # Use lazy initialization to avoid loading overhead if not needed
        self._species_embedding_layer = None
        self._species_embedding_initialized = False
        
        # Cache for team compositions to avoid recalculation
        self._team_cache = {
            'my_team_ids': None,
            'opp_team_ids': None,
            'battle_tag': None  # To detect when battle changes
        }

    def observe(self, battle: AbstractBattle) -> np.ndarray:
        import logging
        logging.debug(f"StateObserver.observe() called with battle: {battle}")
        
        state = []
        # battle が None の場合や、必要な属性がない場合に StateObserver がエラーにならないように、
        # _build_context や _extract が安全にデフォルト値を返す必要があります。
        # Gymnasiumの reset() から初期状態を得る際、Battleオブジェクトがまだ完全に準備できていない可能性も考慮。
        if battle is None:
            # Battle オブジェクトが None の場合、デフォルトの観測値を返すかエラーを出すか。
            # ここでは、デフォルト値で埋めた観測ベクトルを返すことを試みる。
            # ただし、get_observation_dimension() が正確な次元を返すため、
            # この observe(None) が呼ばれるケースは限定的かもしれない。
            # もし呼ばれるなら、デフォルト値で観測ベクトルを構築するロジックが必要。
            # print("Warning: observe called with None battle object. Returning default observation.")
            # return np.zeros(self.get_observation_dimension(), dtype=np.float32) #次元数分の0配列を返すなど
            # あるいは、StateObserverの設計として observe(None) を許容しない場合はエラーを送出してもよい。
            raise ValueError(
                "StateObserver.observe() called with None battle object, which is not supported for actual observation generation."
            )

        context = self._build_context(battle)

        for group, features in self.spec.items():
            for key, meta in features.items():
                raw_default = meta.get("default", 0)
                try:
                    default_val = (
                        eval(raw_default)
                        if isinstance(raw_default, str)
                        else raw_default
                    )
                except Exception:
                    default_val = raw_default

                val = self._extract(meta["battle_path"], context, default_val)

                # self.encoders の初期化は __init__ で行われているはず
                enc_func = self.encoders.get(
                    (group, key),
                    lambda x: (
                        [float(x)] if not isinstance(x, list) else [float(i) for i in x]
                    ),
                )  # デフォルトエンコーダもリストを返すように
                
                # Special handling for move embeddings that return lists directly
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (int, float)):
                    # This is already a numeric list (like move embedding), use it directly
                    encoded_val = [float(x) for x in val]
                else:
                    # Use encoder function
                    encoded_val = enc_func(val)
                    
                    # Handle multi-dimensional features (like move embeddings)
                    if isinstance(encoded_val, list) and any(isinstance(item, list) for item in encoded_val):
                        # Flatten nested lists for move embeddings
                        flattened = []
                        for item in encoded_val:
                            if isinstance(item, list):
                                flattened.extend([float(x) for x in item])
                            else:
                                flattened.append(float(item))
                        encoded_val = flattened

                # デバッグ用printは条件を絞るか、詳細ログレベルで管理した方が良い
                

                state.extend(
                    encoded_val if isinstance(encoded_val, list) else [encoded_val]
                )

        return np.array(state, dtype=np.float32)
    
    def _get_damage_calculator(self) -> DamageCalculator:
        """Get damage calculator with lazy initialization for performance."""
        if not self._damage_calculator_initialized:
            try:
                data_loader = DataLoader()
                self._damage_calculator = DamageCalculator(data_loader)
                print("Damage calculator initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize damage calculator: {e}")
                self._damage_calculator = None
            self._damage_calculator_initialized = True
        return self._damage_calculator
    
    def _get_move_embedding_layer(self):
        """Get move embedding layer with lazy initialization for performance."""
        if not self._move_embedding_initialized:
            try:
                import torch
                from src.agents.move_embedding_layer import MoveEmbeddingLayer
                device = torch.device('cpu')  # Use CPU for state observation
                # Try to load the saved move embeddings
                embedding_file = 'config/move_embeddings_256d_fixed.pkl'
                self._move_embedding_layer = MoveEmbeddingLayer(embedding_file, device)
                print("Move embedding layer initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize move embedding layer: {e}")
                self._move_embedding_layer = None
            self._move_embedding_initialized = True
        return self._move_embedding_layer
    
    def _get_species_embedding_layer(self):
        """Get species embedding layer with lazy initialization for performance."""
        if not self._species_embedding_initialized:
            try:
                import torch
                from src.agents.species_embedding_layer import SpeciesEmbeddingLayer
                device = torch.device('cpu')  # Use CPU for state observation
                self._species_embedding_layer = SpeciesEmbeddingLayer(
                    vocab_size=1026,
                    embed_dim=32,
                    stats_csv_path="config/pokemon_stats.csv",
                    device=device
                )
                print("Species embedding layer initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize species embedding layer: {e}")
                self._species_embedding_layer = None
            self._species_embedding_initialized = True
        return self._species_embedding_layer

    def get_observation_dimension(self) -> int:
        """
        state_spec.yml に基づいて観測ベクトルの総次元数を計算します。
        エンコーダーの出力次元を考慮します。
        """
        dimension = 0
        # エンコーダーが実際に返すリストの長さを確認するために、
        # 各特徴量に対してダミーデータでエンコーダーを一度実行してみるのが最も確実です。
        # このメソッドは初期化時に一度だけ呼ばれる想定なので、多少処理が重くても許容範囲。

        # _build_encodersで作成されたエンコーダ関数を実際に使って次元を計算
        for group, features in self.spec.items():
            for key, meta in features.items():
                # エンコーダを取得
                # self.encodersのキーは (group, key)
                encoder_func = self.encoders.get((group, key))
                if encoder_func is None:
                    # 'identity' や未定義の場合、デフォルトでは1次元と仮定
                    # ただし、デフォルトエンコーダがリストを返す場合もあるので注意
                    # ここでは、デフォルトエンコーダが常に単一のfloatをリストに入れたものを返すと仮定して1とする
                    print(
                        f"Warning: No specific encoder found for {group}.{key}. Assuming 1 dimension."
                    )
                    dimension += 1
                    continue

                # エンコーダにダミーデータ（デフォルト値など）を渡して出力の長さを確認
                # metaからデフォルト値を取得
                raw_default = meta.get("default", 0)
                
                # Handle special case for move embeddings with dimensions specified
                if "dimensions" in meta:
                    dimension += meta["dimensions"]
                    continue
                
                try:
                    # YAMLで '[1,0]' のようにリスト形式で書かれたデフォルト値も評価
                    default_value_for_test = (
                        eval(raw_default)
                        if isinstance(raw_default, str)
                        else raw_default
                    )
                except Exception:
                    default_value_for_test = raw_default  # 数値やNoneなど

                # onehotエンコーダの場合、デフォルト値がエンコード後のリスト形式になっていることがある
                # それ以外の場合、エンコーダが処理できる型（None, int, float, str, Enumなど）のダミー値を渡す
                # 例： 'identity'なら数値、'onehot'ならカテゴリ文字列やNone

                encoder_type = meta.get("encoder", "identity")
                test_input = None  # デフォルトのテスト入力

                if encoder_type == "onehot":
                    # onehot の場合、デフォルト値がエンコード済みリストであるか、
                    # classes の最初の要素、あるいは "none" をテスト入力とする
                    if isinstance(
                        default_value_for_test, list
                    ):  # デフォルトが既にリストならその長さを採用できるが、エンコーダを通す方が確実
                        test_input = (
                            default_value_for_test  # これをエンコーダがどう扱うか
                        )
                    elif meta.get("classes"):
                        test_input = meta["classes"][0]  # 最初のクラスでテスト
                    else:  # classesがないonehotは通常ありえないが、フォールバック
                        test_input = "none"
                elif encoder_type == "linear_scale" or encoder_type == "identity":
                    # 数値を期待するエンコーダには0やデフォルト値（数値化可能なもの）
                    try:
                        test_input = float(
                            default_value_for_test
                            if not isinstance(default_value_for_test, list)
                            else 0
                        )
                    except (ValueError, TypeError):
                        test_input = 0.0  # フォールバック
                else:  # 不明なエンコーダタイプ
                    test_input = default_value_for_test  # そのまま渡してみる

                try:
                    encoded_output = encoder_func(test_input)
                    if isinstance(encoded_output, (list, np.ndarray)):
                        dimension += len(encoded_output)
                    else:  # スカラー値が返ってきた場合 (エンコーダの実装による)
                        dimension += 1
                except Exception as e:
                    print(
                        f"Error while testing encoder for {group}.{key} with input '{test_input}': {e}"
                    )
                    # エラー時はフォールバックとして1次元加算（あるいは設定に基づいてエラーを出す）
                    if meta.get("encoder") == "onehot" and meta.get("classes"):
                        dimension += len(meta.get("classes"))  # onehotならクラス数
                    else:
                        dimension += 1
                    print(
                        f"Warning: Assuming 1 dimension for {group}.{key} due to encoder test error."
                    )

        if dimension == 0:
            raise ValueError(
                "Calculated observation dimension is 0. Check state_spec.yml and StateObserver.get_observation_dimension()."
            )
        return dimension

    def _build_context(self, battle: AbstractBattle) -> dict:
        """Build context with team Pokedex IDs and damage calculation support."""
        ctx = {"battle": battle}
        
        # Get battle identifier for caching
        battle_tag = f"{battle.battle_tag}_{battle.turn}" if hasattr(battle, 'battle_tag') else str(id(battle))
        
        # Build basic context
        my_team = list(battle.team.values()) if battle.team else []
        active = next((p for p in my_team if p.active), None)
        ctx["active"] = active
        ctx["active_sorted_moves"] = (
            sorted(active.moves.values(), key=lambda m: m.id)
            if active and active.moves
            else []
        )

        bench = [p for p in my_team if not p.active] if active else my_team
        ctx["bench1"] = bench[0] if len(bench) > 0 else None
        ctx["bench2"] = bench[1] if len(bench) > 1 else None

        # Add aliases for damage calculation compatibility with move arrays
        def create_pokemon_with_move_array(pokemon):
            """Create a wrapper that allows moves[index] access."""
            if pokemon is None:
                return None
                
            class PokemonWithMoveArray:
                def __init__(self, original_pokemon):
                    # Copy all attributes from original pokemon
                    for attr in dir(original_pokemon):
                        if not attr.startswith('_'):
                            try:
                                setattr(self, attr, getattr(original_pokemon, attr))
                            except:
                                pass  # Skip attributes that can't be copied
                    
                    # Create moves array from moves dict
                    if hasattr(original_pokemon, 'moves') and original_pokemon.moves:
                        moves_list = sorted(original_pokemon.moves.values(), key=lambda m: m.id)
                        # Pad to 4 moves with None
                        self.moves = moves_list + [None] * (4 - len(moves_list))
                    else:
                        self.moves = [None, None, None, None]
            
            return PokemonWithMoveArray(pokemon)
        
        ctx["my_active"] = create_pokemon_with_move_array(active)
        ctx["my_bench1"] = create_pokemon_with_move_array(ctx["bench1"])
        ctx["my_bench2"] = create_pokemon_with_move_array(ctx["bench2"])

        # Use teampreview_opponent_team during team preview, opponent_team during battle
        import logging
        if hasattr(battle, 'teampreview_opponent_team') and battle.teampreview_opponent_team:
            opp_team_list = list(battle.teampreview_opponent_team)
            logging.debug(f"Using teampreview_opponent_team: {len(opp_team_list)} Pokemon")
        else:
            opp_team_list = list(battle.opponent_team.values()) if battle.opponent_team else []
            logging.debug(f"Using opponent_team: {len(opp_team_list)} Pokemon")
        self.opp_total_estimate = max(self.opp_total_estimate, len(opp_team_list))
        opp_alive_seen = sum(1 for p in opp_team_list if not p.fainted)
        unknown_remaining = max(0, self.opp_total_estimate - len(opp_team_list))

        opp_active = next((p for p in opp_team_list if p.active), None)
        ctx["opp_active"] = opp_active
        opp_bench = [p for p in opp_team_list if not p.active] if opp_active else opp_team_list
        ctx["opp_bench1"] = opp_bench[0] if len(opp_bench) > 0 else None
        ctx["opp_bench2"] = opp_bench[1] if len(opp_bench) > 1 else None
        ctx["my_alive_count"] = sum(1 for p in my_team if not p.fainted)
        ctx["opp_alive_count"] = opp_alive_seen + unknown_remaining

        # Create opp_team array for damage calculation (6-element array with None padding)
        opp_team_array = []
        for i in range(6):
            if i < len(opp_team_list):
                opp_team_array.append(create_pokemon_with_move_array(opp_team_list[i]))
            else:
                opp_team_array.append(None)
        ctx["opp_team"] = opp_team_array
        
        ctx["type_matchup_vec"] = self.type_matchup_extractor.extract(battle)

        # Add team Pokedex IDs with caching for efficiency
        if self._team_cache['battle_tag'] != battle_tag:
            # Cache miss - recalculate team IDs
            my_team_ids = self.species_mapper.get_team_pokedex_ids(my_team)
            opp_team_ids = self.species_mapper.get_team_pokedex_ids(opp_team_list)
            
            self._team_cache.update({
                'my_team_ids': my_team_ids,
                'opp_team_ids': opp_team_ids,
                'battle_tag': battle_tag
            })
        
        # Add team information to context
        ctx["my_team"] = my_team
        ctx["opp_team"] = opp_team_list
        
        # Add individual team member Pokedex IDs for easy access
        for i in range(6):
            ctx[f"my_team{i+1}_pokedex_id"] = self._team_cache['my_team_ids'][i]
            ctx[f"opp_team{i+1}_pokedex_id"] = self._team_cache['opp_team_ids'][i]
        
        # Add damage calculation function to context
        damage_calc = self._get_damage_calculator()
        if damage_calc:
            # Cache for function results within single observation
            damage_cache = {}
            
            # Create a wrapper function for damage calculation that's accessible from eval()
            def calc_damage_expectation_for_ai(attacker, target, move, terastallized=False):
                
                # Validate basic inputs - raise error if attacker or target are None/invalid
                if not attacker or not target:
                    raise ValueError(f"Invalid input: attacker={attacker}, target={target}, move={move}")
                
                # Handle case where move is None (e.g., Pokemon has fewer than 4 moves)
                # Return safe default values: 0% expected damage, 0% variance
                if not move:
                    return (0.0, 0.0)
                
                # Get target name and move name for error messages and caching
                target_name = target.species if hasattr(target, 'species') else str(target)
                move_name = move.id if hasattr(move, 'id') else str(move)
                
                # Create cache key from function parameters
                attacker_id = id(attacker)
                target_id = id(target)
                move_id = id(move)
                cache_key = (attacker_id, target_id, move_id, terastallized)
                
                # Return cached result if available
                if cache_key in damage_cache:
                    return damage_cache[cache_key]
                
                # Extract attacker stats
                attacker_stats = {
                    'level': getattr(attacker, 'level', 50),
                    'attack': attacker.stats.get('atk', 100),
                    'special_attack': attacker.stats.get('spa', 100),
                    'speed': attacker.stats.get('spe', 100),
                    'weight': getattr(attacker, 'weight', 100),
                    'rank_attack': attacker.boosts.get('atk', 0),
                    'rank_special_attack': attacker.boosts.get('spa', 0),
                    'type1': attacker.type_1.name if attacker.type_1 else 'Normal',
                    'type2': attacker.type_2.name if attacker.type_2 else None,
                    'tera_type': attacker.tera_type.name if hasattr(attacker, 'tera_type') and attacker.tera_type else None,
                    'is_terastalized': terastallized
                }
                
                # Get move type
                move_type = move.type.name if hasattr(move, 'type') and move.type else 'Normal'
                
                # Check if move is STATUS type before damage calculation
                from poke_env.environment.move_category import MoveCategory
                if hasattr(move, 'category') and move.category == MoveCategory.STATUS:
                    # STATUS moves (like protect, rest) don't deal damage
                    result = (0.0, 0.0)
                elif hasattr(move, 'base_power') and (move.base_power is None or move.base_power == 0):
                    # Moves with no base power don't deal damage
                    result = (0.0, 0.0)
                else:
                    # Call DamageCalculator for actual damage-dealing moves
                    result = damage_calc.calculate_damage_expectation_for_ai(
                        attacker_stats, target, move, move_type
                    )
                
                # Cache result for future use within this observation
                damage_cache[cache_key] = result
                return result
            
            ctx["calc_damage_expectation_for_ai"] = calc_damage_expectation_for_ai
        
        # Add move embedding functionality to context
        move_embedding_layer = self._get_move_embedding_layer()
        if move_embedding_layer:
            # Create a wrapper class to provide convenient move embedding access
            class MoveEmbeddingProvider:
                def __init__(self, embedding_layer):
                    self.embedding_layer = embedding_layer
                    self.embedding_cache = {}  # Cache embeddings within a single observation
                
                def get_move_embedding(self, move_id):
                    """Get 256-dimensional embedding for a move by its ID."""
                    if move_id is None:
                        # Return zero vector for missing moves
                        return [0.0] * 256
                    
                    # Check cache first
                    if move_id in self.embedding_cache:
                        return self.embedding_cache[move_id]
                    
                    try:
                        # Get embedding from layer
                        embedding_tensor = self.embedding_layer.get_move_embedding(move_id)
                        if embedding_tensor is not None:
                            # Convert to list and cache
                            embedding_list = embedding_tensor.detach().cpu().numpy().tolist()
                            self.embedding_cache[move_id] = embedding_list
                            return embedding_list
                        else:
                            # Move not found, return zero vector
                            zero_embedding = [0.0] * 256
                            self.embedding_cache[move_id] = zero_embedding
                            return zero_embedding
                    except Exception as e:
                        print(f"Warning: Failed to get embedding for move {move_id}: {e}")
                        # Return zero vector on error
                        zero_embedding = [0.0] * 256
                        self.embedding_cache[move_id] = zero_embedding
                        return zero_embedding
            
            ctx["move_embedding"] = MoveEmbeddingProvider(move_embedding_layer)
        else:
            # Fallback provider that returns zero vectors
            class FallbackMoveEmbeddingProvider:
                def get_move_embedding(self, move_id):
                    return [0.0] * 256
            
            ctx["move_embedding"] = FallbackMoveEmbeddingProvider()
        
        # Add species embedding functionality to context
        species_embedding_layer = self._get_species_embedding_layer()
        if species_embedding_layer:
            # Create a wrapper class to provide convenient species embedding access
            class SpeciesEmbeddingProvider:
                def __init__(self, embedding_layer):
                    self.embedding_layer = embedding_layer
                    self.embedding_cache = {}  # Cache embeddings within a single observation
                
                def get_species_embedding(self, species_id):
                    """Get normalized embedding for a Pokemon species ID."""
                    if species_id in self.embedding_cache:
                        return self.embedding_cache[species_id]
                    
                    try:
                        if species_id is None or species_id == 0:
                            # Return zero vector for unknown species
                            zero_embedding = [0.0] * 32
                            self.embedding_cache[species_id] = zero_embedding
                            return zero_embedding
                        
                        # Get embedding from layer
                        embedding = self.embedding_layer.get_species_embedding(int(species_id))
                        embedding_list = embedding.detach().cpu().numpy().tolist()
                        
                        self.embedding_cache[species_id] = embedding_list
                        return embedding_list
                    except Exception as e:
                        print(f"Warning: Failed to get species embedding for ID {species_id}: {e}")
                        # Return zero vector on error
                        zero_embedding = [0.0] * 32
                        self.embedding_cache[species_id] = zero_embedding
                        return zero_embedding
            
            ctx["species_embedding"] = SpeciesEmbeddingProvider(species_embedding_layer)
        else:
            # Fallback provider that returns zero vectors
            class FallbackSpeciesEmbeddingProvider:
                def get_species_embedding(self, species_id):
                    return [0.0] * 32
            
            ctx["species_embedding"] = FallbackSpeciesEmbeddingProvider()

        return ctx

    def _extract(self, path: str, ctx: dict, default):
        """Extract feature value from context using the specified path."""
        try:
            # pathがNoneや空文字の場合も考慮
            if not path:
                return default
            
            # Special handling for team Pokedex ID access
            if path.endswith('.species_id'):
                # Direct access to cached Pokedex IDs
                if 'my_team[0].species_id' in path:
                    return ctx.get('my_team1_pokedex_id', 0)
                elif 'my_team[1].species_id' in path:
                    return ctx.get('my_team2_pokedex_id', 0)
                elif 'my_team[2].species_id' in path:
                    return ctx.get('my_team3_pokedex_id', 0)
                elif 'my_team[3].species_id' in path:
                    return ctx.get('my_team4_pokedex_id', 0)
                elif 'my_team[4].species_id' in path:
                    return ctx.get('my_team5_pokedex_id', 0)
                elif 'my_team[5].species_id' in path:
                    return ctx.get('my_team6_pokedex_id', 0)
                elif 'opp_team[0].species_id' in path:
                    return ctx.get('opp_team1_pokedex_id', 0)
                elif 'opp_team[1].species_id' in path:
                    return ctx.get('opp_team2_pokedex_id', 0)
                elif 'opp_team[2].species_id' in path:
                    return ctx.get('opp_team3_pokedex_id', 0)
                elif 'opp_team[3].species_id' in path:
                    return ctx.get('opp_team4_pokedex_id', 0)
                elif 'opp_team[4].species_id' in path:
                    return ctx.get('opp_team5_pokedex_id', 0)
                elif 'opp_team[5].species_id' in path:
                    return ctx.get('opp_team6_pokedex_id', 0)
                    
            return eval(
                path,
                {
                    "PokemonType": PokemonType,
                    "MoveCategory": MoveCategory,
                    "AbstractBattle": AbstractBattle,
                },
                ctx,
            )  # Enum型をevalのスコープに追加
        except (AttributeError, TypeError, IndexError, NameError, SyntaxError) as e:
            import logging
            # 詳細なエラー情報をログ出力してからエラーを再発生させる
            logging.error(f"EXTRACT ERROR: Path '{path}' failed with {type(e).__name__}: '{e}'")
            logging.error(f"EXTRACT ERROR: Context keys: {list(ctx.keys())}")
            
            # Additional debugging for specific problematic paths
            if 'my_bench2' in path:
                logging.error(f"EXTRACT ERROR: my_bench2 = {ctx.get('my_bench2')}")
                if ctx.get('my_bench2') is not None:
                    bench2 = ctx.get('my_bench2')
                    logging.error(f"EXTRACT ERROR: my_bench2.moves = {getattr(bench2, 'moves', 'NO_MOVES_ATTR')}")
                    if hasattr(bench2, 'moves'):
                        logging.error(f"EXTRACT ERROR: my_bench2.moves length = {len(bench2.moves)}")
            
            if 'opp_team' in path:
                logging.error(f"EXTRACT ERROR: opp_team = {ctx.get('opp_team')}")
                opp_team = ctx.get('opp_team')
                if opp_team is not None:
                    logging.error(f"EXTRACT ERROR: opp_team length = {len(opp_team)}")
                    for i, opp in enumerate(opp_team):
                        logging.error(f"EXTRACT ERROR: opp_team[{i}] = {opp}")
                    # Additional debugging for the specific failing path
                    if 'opp_team[1]' in path:
                        logging.error(f"EXTRACT ERROR: Attempting to access opp_team[1], but opp_team length = {len(opp_team)}")
                        logging.error(f"EXTRACT ERROR: Available indices: 0 to {len(opp_team)-1}")
            
            # tera_type関連のエラーの詳細ログ
            if 'tera_type' in path:
                logging.error(f"EXTRACT ERROR: tera_type path detected")
                if 'battle.active_pokemon' in path:
                    battle = ctx.get('battle')
                    if battle:
                        logging.error(f"EXTRACT ERROR: battle.active_pokemon = {getattr(battle, 'active_pokemon', 'NO_ACTIVE_POKEMON')}")
                        if hasattr(battle, 'active_pokemon') and battle.active_pokemon:
                            active_pokemon = battle.active_pokemon
                            logging.error(f"EXTRACT ERROR: active_pokemon.tera_type = {getattr(active_pokemon, 'tera_type', 'NO_TERA_TYPE')}")
                            logging.error(f"EXTRACT ERROR: active_pokemon attributes: {[attr for attr in dir(active_pokemon) if not attr.startswith('_')]}")
                        else:
                            logging.error(f"EXTRACT ERROR: battle.active_pokemon is None or missing")
                    else:
                        logging.error(f"EXTRACT ERROR: battle is None")
                        
                if 'battle.opponent_active_pokemon' in path:
                    battle = ctx.get('battle')
                    if battle:
                        logging.error(f"EXTRACT ERROR: battle.opponent_active_pokemon = {getattr(battle, 'opponent_active_pokemon', 'NO_OPP_ACTIVE')}")
                        if hasattr(battle, 'opponent_active_pokemon') and battle.opponent_active_pokemon:
                            opp_active = battle.opponent_active_pokemon
                            logging.error(f"EXTRACT ERROR: opponent_active_pokemon.tera_type = {getattr(opp_active, 'tera_type', 'NO_TERA_TYPE')}")
                        else:
                            logging.error(f"EXTRACT ERROR: battle.opponent_active_pokemon is None or missing")
            
            # Handle opponent bench Pokemon that may be None (not yet revealed)
            if isinstance(e, AttributeError) and ('opp_bench' in path or 'battle.opponent_active_pokemon' in path):
                logging.debug(f"Opponent Pokemon not available, using default for: {path}")
                return default
            
            # IndexErrorは詳細ログ出力後にエラーを再発生（これが目標）
            raise e
        except Exception as e:  # その他の予期せぬエラー
            import logging
            logging.error(f"EXTRACT UNEXPECTED ERROR: Path '{path}' failed with {type(e).__name__}: '{e}'")
            # エラーを再発生させる（フォールバックせずに停止）
            raise e

    def _build_encoders(self, spec: dict):
        # (既存の _build_encoders メソッドは概ねそのままで良いが、出力が常にリストになるように調整を検討)
        # onehotエンコーダのデフォルト値の扱いを修正
        enc = {}
        for group, features in spec.items():
            for key, meta in features.items():
                enc_key = (group, key)
                kind = meta.get("encoder", "identity")
                # default値は eval せずに文字列のまま _onehot_encoder_simple に渡す
                raw_default_for_lambda = meta.get("default", "0")  # evalしない

                if kind == "identity":
                    # 出力をリスト[float]に統一
                    enc[enc_key] = lambda x, d=raw_default_for_lambda: [
                        float(
                            x if x is not None else eval(d) if isinstance(d, str) else d
                        )
                    ]
                elif kind == "onehot":
                    classes = meta.get("classes", [])
                    # _onehot_encoder_simple を呼び出すlambda
                    enc[enc_key] = (
                        lambda val, cls_list=classes, default_str=raw_default_for_lambda: self._onehot_encoder_simple(
                            val, cls_list, default_str
                        )
                    )
                elif kind == "linear_scale":
                    lo, hi = meta.get("range", [0, 1])
                    # evalできる形式のデフォルト値文字列を期待
                    enc[enc_key] = (
                        lambda x, lower=lo, upper=hi, d_str=raw_default_for_lambda: [
                            (
                                ((float(x) - lower) / (upper - lower))
                                if x is not None and upper > lower
                                else (
                                    float(eval(d_str))
                                    if isinstance(d_str, str)
                                    else float(d_str)
                                )
                            )
                        ]
                    )
                else:  # 未知のエンコーダタイプもリスト[float]で返す
                    enc[enc_key] = lambda x, d=raw_default_for_lambda: [
                        float(
                            x if x is not None else eval(d) if isinstance(d, str) else d
                        )
                    ]
        return enc

    # _onehot_encoder_simple を StateObserver のメソッドとして定義
    def _onehot_encoder_simple(self, val, cls_list: list, default_str_val: str) -> list:
        """onehotエンコーディングを行い、結果をリストで返す"""
        value_to_encode_str = ""
        if val is None:
            value_to_encode_str = "none"
        elif isinstance(val, (PokemonType, MoveCategory)):  # Enum型の場合
            value_to_encode_str = val.name.lower()
        elif hasattr(val, "name") and isinstance(
            val.name, str
        ):  # 他のEnumっぽいオブジェクト
            value_to_encode_str = val.name.lower()
        else:
            value_to_encode_str = str(val).lower()

        if not cls_list:  # クラスリストが空の場合
            # default_str_valがエンコード済みリスト形式（例: '[0,0,1]'）ならそれを評価して返す
            try:
                default_as_list = eval(default_str_val)
                if isinstance(default_as_list, list):
                    return [float(i) for i in default_as_list]
            except Exception:
                pass
            return []  # 空のクラスリストに対するエンコードは空リスト

        encoded_list = [
            1.0 if value_to_encode_str == str(c).lower() else 0.0 for c in cls_list
        ]

        if sum(encoded_list) == 0:  # クラスリストにない値が来た場合
            try:
                default_as_list = eval(default_str_val)
                if isinstance(default_as_list, list) and len(default_as_list) == len(
                    cls_list
                ):
                    return [float(i) for i in default_as_list]
                # default_str_val が単一の値で、それがクラスリストにあればonehot化する試みも可能だが、
                # ここでは default_str_val がエンコード済みリストであることを期待する。
            except Exception:
                # evalに失敗した場合やリストでない場合、'none'を探す
                if "none" in [str(c).lower() for c in cls_list]:
                    return [1.0 if "none" == str(c).lower() else 0.0 for c in cls_list]
                # 'none'もなければ、すべて0のリストを返す
        return encoded_list
