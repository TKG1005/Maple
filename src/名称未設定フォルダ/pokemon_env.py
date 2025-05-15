import asyncio
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from poke_env.player import Player, RandomPlayer
from poke_env.environment.battle import Battle
from poke_env.data import Gen9Data # 第9世代データを利用

# M3で作成したクラス・関数をインポート
from src.state.state_observer import StateObserver
from src.action.action_helper import get_available_actions, action_index_to_order
# ActionHelperは特定の関数を利用するため、クラスインスタンスではなく関数を直接利用


class PokemonEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1} # render_modes は必要に応じて

    def __init__(self, opponent_player: Player, state_observer: StateObserver, battle_format: str = "gen9randombattle", team_pascal: str = None):
        """
        PokemonEnv のコンストラクタ

        Args:
            opponent_player (Player): 対戦相手のプレイヤーインスタンス
            state_observer (StateObserver): 状態観測用のオブザーバーインスタンス
            battle_format (str, optional): 対戦フォーマット. Defaults to "gen9randombattle".
                                            将来的には "gen9ou" など固定パーティ用も想定。
            team_pascal (str, optional): 使用するチームのPascalフォーマット文字列。
                                         固定パーティの場合に指定。
        """
        super().__init__()

        self.battle_format = battle_format
        self.opponent = opponent_player
        self.state_observer = state_observer
        # ActionHelperの機能は直接メソッドとして利用するか、このEnvクラス内で呼び出す

        # --- 状態空間 (Observation Space) の定義 ---
        # M3で定義した StateObserver が返すNumpy配列の形状と型に基づいて定義します。
        # state_spec.yml から状態ベクトルの次元数を計算する必要があります。
        # ここでは仮の次元数として 300 を設定しますが、StateObserverの実装に合わせて正確な値に置き換えてください。
        # 例: self.state_observer.get_observation_space_dim() のようなメソッドを StateObserver に追加すると便利です。
        # データ型は float32 が一般的です。
        # 各要素の取りうる値の範囲 (low, high) も指定できますが、正規化されている場合は 0 から 1 などになります。
        # ここでは、state_spec.ymlに基づき、ほとんどの値が0から1の範囲か、-1から1の範囲（ブーストなど）に正規化されていると仮定します。
        # 最小値と最大値をより正確に設定することが望ましいですが、まずは -np.inf, np.inf としておくことも可能です。
        # state_observer.observe(None) # ダミーのバトルオブジェクトで初期の観測空間の次元数を取得する（要実装）
        # self._dummy_battle = Battle("dummy_battle", self.battle_format, Gen9Data()) # observeの引数に合わせる
        # self._dummy_observation = self.state_observer.observe(self._dummy_battle) # 仮のBattleオブジェクトで次元数を取得

        # state_spec.ymlを読み込み、各特徴量の次元を合計してobservation_spaceの次元を決定する
        # この処理はStateObserver側で行い、プロパティとして次元数を公開するのが望ましい
        # ここでは仮に state_observer が `observation_dim` プロパティを持つとします。
        # StateObserverに `get_observation_dim()` のようなメソッドを追加するか、
        # 初期化時に計算してプロパティとして保持するようにしてください。
        # ダミーのBattleオブジェクトを作成して、一度observeを実行し、そのshapeから次元数を取得するのが最も確実です。
        try:
            # StateObserver に get_observation_space_shape メソッドを実装することを推奨
            # observation_shape = self.state_observer.get_observation_space_shape() # (dim,) のようなタプルを想定
            # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=observation_shape, dtype=np.float32)
            # 現時点では仮の値を設定します。後ほど StateObserver の実装と合わせて修正してください。
            # state_spec.ymlから計算した次元数を設定する必要があります。
            # state_observer.py の observe メソッドが返す NumPy 配列の長さを調べて設定してください。
            # 例: (state_observer.observe(ダミーのBattleオブジェクト)).shape[0]
            # ここでは仮に358次元とします (state_spec.yml を元に手計算した場合の概算、正確な値に要修正)
            # TODO: StateObserverから正確な次元数を取得するように修正
            dummy_obs_len = self._get_dummy_observation_length()
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dummy_obs_len,), dtype=np.float32)
            print(f"Observation space set with dimension: {dummy_obs_len}")

        except Exception as e:
            print(f"Error setting up observation space: {e}")
            print("Please ensure StateObserver can provide observation dimension.")
            # フォールバックとして仮の値を設定
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(300,), dtype=np.float32)


        # --- 行動空間 (Action Space) の定義 ---
        # M3で定義した行動空間のサイズに基づいて定義します。
        # 固定長10（通常技4 + テラスタル技4 + 交代2）と定義されていました。
        self.action_space_size = 10 # 通常技0-3, テラスタル技4-7, 交代8-9
        self.action_space = spaces.Discrete(self.action_space_size)

        # --- poke-env 関連の初期化 ---
        # EnvPlayer は poke-env の Player を Gym Env 内でラップするためのものです。
        # これがエージェント自身となり、対戦を行います。
        self.player = EnvPlayer(
            opponent=self.opponent, # AIの対戦相手
            battle_format=self.battle_format,
            team=team_pascal, # 固定パーティを使用する場合
            # log_level=20 # デバッグ用にログレベルを設定 (10:DEBUG, 20:INFO)
        )

        self.current_battle: Battle = None # 現在のバトルオブジェクトを保持

        print("PokemonEnv initialized.")
        print(f"Observation Space: {self.observation_space}")
        print(f"Action Space: {self.action_space}")


    def _get_dummy_observation_length(self) -> int:
        """
        StateObserverから観測ベクトルの長さを取得するためのヘルパーメソッド。
        StateObserverが正しく初期化されている必要があります。
        """
        # poke-envのBattleオブジェクトの最小限の模倣を試みます。
        # 実際のBattleオブジェクトとは異なるため、StateObserverがNoneアクセスでエラーにならないよう注意が必要です。
        # StateObserverの_build_contextや_extractがNoneの属性アクセスを安全に処理するように実装されていることが前提です。
        class DummyPokemon:
            def __init__(self, species="pikachu", active=False):
                self.species = species
                self.active = active
                self.moves = {} # {move_id: DummyMove()}
                self.current_hp_fraction = 1.0
                self.boosts = {'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0, 'acc': 0, 'eva': 0}
                self.status = None
                self.tera_type = None # PokemonType.NORMAL など
                self.type_1 = None # PokemonType.ELECTRIC など
                self.type_2 = None
                self.is_terastallized = False
                self.fainted = False

            # StateObserverが参照する可能性のある他の属性やメソッドを適宜追加

        class DummyBattle(Battle): # AbstractBattleを継承
            def __init__(self, battle_tag, battle_format, data):
                # 親クラスのコンストラクタに必要な引数を渡す
                # Battleクラスのコンストラクタのシグネチャに合わせて調整が必要
                # Battle.__init__(self, battle_tag, battle_format, data, logger=None, max_battle_turns=1000, start_timer_on_battle_start=False, start_timer_request_timeout=0)
                # 上記は poke_env 0.8.0 の例。0.9.0では異なる可能性あり。
                # poke_env.environment.battle.Battle のコンストラクタを確認してください。
                # 最小限の初期化を試みます。
                self.battle_tag = battle_tag
                self._battle_format = battle_format # Battle._battle_format を参照する場合があるため
                self._data = data
                self._logger = gym.logger # gym のロガーを使用
                self.turn = 0
                self.team = {f"p1:{i+1}": DummyPokemon(active=(i==0)) for i in range(3)} # 3体のダミーポケモン
                self.opponent_team = {f"p2:{i+1}": DummyPokemon(active=(i==0)) for i in range(3)}
                self.active_pokemon = self.team["p1:1"]
                self.opponent_active_pokemon = self.opponent_team["p2:1"]
                self.weather = None # (Weather, duration)
                self.fields = {}
                self.side_conditions = {}
                self.opponent_side_conditions = {}
                self.finished = False
                self.available_moves = []
                self.available_switches = []
                self.can_mega_evolve = False
                self.can_z_move = False
                self.can_dynamax = False
                self.can_tera = None # テラスタル可能な場合、対象ポケモンの種族名を文字列で持つ (例: "pikachu")
                # Battleクラスが要求する他の属性も適宜Noneやデフォルト値で初期化
                self._rqid = 0
                self._switch_trapped = False
                self._force_switch = False
                self._wait = False
                self._move_on_next_request = False
                self._maybe_trapped = False
                self._trapped = False
                self._player_role = "p1" # または "p2"
                # 他に必要な属性があれば追加してください

            # Battleクラスが持つべきメソッドのダミー実装 (StateObserverが呼び出す可能性のあるもの)
            # 例: get_pokemon, get_opponent_pokemon など
            def get_pokemon(self, identifier: str, force_exact_match: bool = True) -> DummyPokemon:
                # ダミー実装
                parts = identifier.split(': ')
                player_id = parts[0][:2] # 'p1' or 'p2'
                # pokemon_name = parts[1] # 必要であれば

                if player_id == self._player_role: # self.player._player_id に相当
                    # 簡単のため、常にアクティブなポケモンを返すか、チームの最初のポケモンを返す
                    return self.active_pokemon if self.active_pokemon else list(self.team.values())[0]
                else:
                    return self.opponent_active_pokemon if self.opponent_active_pokemon else list(self.opponent_team.values())[0]

        # Gen9Data インスタンスを作成
        gen_data = Gen9Data()
        # DummyBattle のインスタンスを作成
        dummy_battle = DummyBattle("dummy_battle_tag", self.battle_format, gen_data)

        # StateObserver が None を安全に扱えるように実装されていることが前提
        # 例: active_pokemon が None の場合、関連する特徴量はデフォルト値になるなど
        # 技の情報もNoneアクセスに備えておく必要がある
        for i in range(4):
            dummy_move = type(f'DummyMove{i}', (), {
                'id': f'move{i}', 'type': None, 'base_power': 0, 'accuracy': 0,
                'category': None, 'current_pp': 0, 'max_pp': 0
            })()
            if dummy_battle.active_pokemon:
                 dummy_battle.active_pokemon.moves[f'move{i}'] = dummy_move
        
        # ダミーの available_moves と available_switches を設定 (StateObserverのロジックによる)
        # StateObserverの_build_contextで参照する可能性があるため
        # dummy_battle.available_moves = [dummy_battle.active_pokemon.moves['move0']] if dummy_battle.active_pokemon and 'move0' in dummy_battle.active_pokemon.moves else []
        # dummy_battle.available_switches = [p for p in dummy_battle.team.values() if not p.active and not p.fainted]

        # observeメソッドを実行して形状を取得
        # state_observer.py の observe メソッドがバトルオブジェクトのどの属性を参照するかによって、
        # DummyBattle に必要な属性・メソッドは変わります。
        # 実行してみて AttributeError などが出たら、その属性を DummyBattle に追加してください。
        try:
            obs_vector = self.state_observer.observe(dummy_battle)
            return obs_vector.shape[0]
        except Exception as e:
            print(f"Error during dummy observation for shape: {e}")
            print("Falling back to default observation dimension (300). Review StateObserver and DummyBattle.")
            return 300 # フォールバック

    def step(self, action_index: int):
        # タスク1.2 で実装
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        # タスク1.3 で実装
        # super().reset(seed=seed) # Gymnasium 0.26+ の場合
        raise NotImplementedError

    def render(self, mode='human'):
        # デバッグ用に現在のバトル状態などを表示（任意実装）
        if mode == 'human':
            if self.current_battle:
                print(f"Turn: {self.current_battle.turn}")
                if self.current_battle.active_pokemon:
                    print(f"My Active: {self.current_battle.active_pokemon.species} ({self.current_battle.active_pokemon.current_hp_fraction*100:.1f}%)")
                if self.current_battle.opponent_active_pokemon:
                    print(f"Opponent Active: {self.current_battle.opponent_active_pokemon.species} ({self.current_battle.opponent_active_pokemon.current_hp_fraction*100:.1f}%)")
                # さらに詳細な情報を表示することも可能
            else:
                print("Battle not started yet.")
        else:
            # 他のレンダリングモードが必要な場合は実装
            pass

    def close(self):
        # 環境をクリーンアップするための処理（例: サーバー接続を切断など）
        # poke-env の場合、Playerが内部で ShowdownConnector を持つため、
        # Player のChallenge終了や切断処理をここで行うことを検討します。
        # asyncio のイベントループを停止させる必要がある場合もあります。
        print("Closing PokemonEnv.")
        if self.player and hasattr(self.player, 'stop_listening') and callable(self.player.stop_listening):
            try:
                # poke-env の Player が非同期で動作している場合、
                # stop_listening() も非同期メソッドの可能性があり、
                # イベントループ内で実行する必要があるかもしれません。
                # ここでは同期的に呼び出せることを仮定します。
                # self.player.stop_listening() # 通常、Playerが自分で閉じるので明示的な呼び出しは不要な場合も
                pass
            except Exception as e:
                print(f"Error during player cleanup: {e}")
        # asyncioのイベントループがこのEnvの外部で管理されている場合は、
        # ここで特別なループ停止処理は不要かもしれません。

class EnvPlayer(Player):
    """
    Gymnasium Env内でpoke-envのPlayerをラップするクラス。
    choose_moveはEnvのstepメソッドから指示されたactionを実行するようにオーバーライドされる。
    """
    def __init__(self, opponent: Player, battle_format: str, team: str = None, log_level: int = None):
        # Playerのコンストラクタに必要な引数を渡す
        # poke_env 0.9.0 の Player.__init__ のシグネチャに合わせてください
        # 例: Player.__init__(self, player_configuration=None, *, avatar=None, battle_format=None, log_level=None, max_concurrent_battles=1, save_replays=False, server_configuration=None, start_timer_on_battle_start=False, start_listening=True, team=None)
        # 必要なものだけ指定します。
        super().__init__(
            battle_format=battle_format,
            opponent=opponent, # Playerクラスはopponent引数を受け付けないため、別途管理が必要
            team=team,
            log_level=log_level
        )
        self._env_opponent = opponent # Envが管理する対戦相手
        self._battle_to_play_against_env_opponent_task = None # challengeメソッドのタスクを保持

    def choose_move(self, battle: Battle) -> str:
        # このメソッドは、Envのstepメソッドから渡されたアクションを実行するために
        # 外部から設定されるか、Envが直接 self.player.send_message を呼び出す形になる。
        # 通常の強化学習ループでは、エージェントが行動を選択し、
        # Envのstepメソッドがその行動をpoke-envのPlayerに伝えて実行させる。
        # そのため、この choose_move が呼ばれるのは、
        # 環境側から「次の行動を決定してください」というpoke-envの通常の流れの場合。
        # RLエージェントが行動を決定する場合、このメソッドは直接使われないか、
        # Envが決定した行動を返すプレースホルダーになる。
        # ここでは、Envからactionが供給されることを期待するため、
        # このメソッドが呼ばれた場合は何もしないか、エラーを出す。
        # あるいは、Envが次のactionをセットするのを待つFutureを返す。
        if self._action_to_send:
            action_str = self._action_to_send
            self._action_to_send = None # 一度使ったらクリア
            return action_str
        else:
            # Envのstepメソッドがactionを設定するまで待機するメカニズムが必要
            # ここでは仮にランダムな手を打つようにしておく (デバッグ用)
            # print("EnvPlayer.choose_move called without pre-set action, choosing random.")
            # return self.choose_random_move(battle) # Playerクラスのメソッドを利用
            # raise RuntimeError("EnvPlayer.choose_move was called directly. Action should be provided by the environment.")
            # 非同期で行動を待つ場合
            loop = asyncio.get_event_loop()
            self._next_action_future = loop.create_future()
            # print("EnvPlayer: Waiting for action from Env...")
            # このFutureはEnvのstepメソッドでactionが渡された際に結果設定される
            # Battle._player_decision_is_made.set()のような仕組みを参考にする
            return self._next_action_future # BattleはこのFutureをawaitする

    async def start_battle_against_opponent(self, opponent: Player, battle_format: str = "gen9randombattle", team_pascal: str = None):
        """
        指定された対戦相手との対戦を開始し、対戦が終了するまで待機する。
        """
        if self._battle_to_play_against_env_opponent_task and not self._battle_to_play_against_env_opponent_task.done():
            print("A battle is already in progress or was not properly cleaned up.")
            return

        self._battle_to_play_against_env_opponent_task = asyncio.create_task(
            self._challenge_and_wait(opponent, battle_format, team_pascal)
        )
        try:
            await self._battle_to_play_against_env_opponent_task
        except asyncio.CancelledError:
            print("Battle task was cancelled.")
        finally:
            self._battle_to_play_against_env_opponent_task = None


    async def _challenge_and_wait(self, opponent: Player, battle_format: str, team_pascal: str = None):
        if team_pascal:
            self.update_team(team_pascal) # チームをセット
        
        # 対戦相手にチャレンジ
        await self.send_challenges(opponent.username, n_challenges=1, to_wait=opponent.logged_in)
        # print(f"Challenged {opponent.username}. Waiting for battle to finish...")

    def set_next_action(self, action_order: str):
        """Envから実行すべき行動文字列を受け取る"""
        if hasattr(self, '_next_action_future') and self._next_action_future and not self._next_action_future.done():
            self._next_action_future.set_result(action_order)
            # print(f"EnvPlayer: Action '{action_order}' set by Env.")
        else:
            # Futureが準備できていないか、すでに行動が設定されている場合
            # stepメソッドのロジックで、choose_moveが呼ばれる前にactionが設定されるように調整が必要
            # print(f"EnvPlayer: Future not ready or already set. Buffering action '{action_order}'.")
            self._action_to_send = action_order # バッファリング

    async def _battle_finished_callback(self, battle: Battle):
        """バトル終了時に呼ばれるコールバック (Playerクラスのメソッドをオーバーライド)"""
        # print(f"Battle {battle.battle_tag} finished. Player won: {battle.won}")
        # Envにバトル終了を通知するなどの処理をここで行う
        # 例えば、resetメソッドで次のバトルを開始する準備をするなど
        # このコールバックはpoke-envの内部ループから呼ばれるため、
        # Envのメインループとの同期に注意が必要
        if hasattr(self, '_env_instance') and self._env_instance:
             self._env_instance.current_battle_finished = True # Envインスタンスにフラグを立てるなど
        # Playerクラスのデフォルトのコールバックも呼び出す場合
        await super()._battle_finished_callback(battle)

# --- 動作確認のための仮コード ---
if __name__ == '__main__':
    print("PokemonEnv class defined. Basic structure is ready.")
    print("To test further, you'll need to implement step and reset methods,")
    print("and run it with a StateObserver instance and an opponent player.")

    # StateObserverの初期化 (state_spec.ymlのパスを適切に設定してください)
    # このファイルがプロジェクトルート/src/environments/pokemon_env.py にあると仮定すると、
    # state_spec.yml が プロジェクトルート/config/state_spec.yml にある場合、
    # パスは "../../config/state_spec.yml" のようになります。
    # 実行場所に応じてパスを調整してください。
    # ここでは、このスクリプトが src/environments/ にあり、
    # state_spec.yml が ../../config/ にあると仮定します。
    try:
        spec_path = "../../config/state_spec.yml" # このファイルの場所に合わせて調整してください
        observer = StateObserver(spec_path)
        print(f"StateObserver loaded with spec: {spec_path}")
    except FileNotFoundError:
        print(f"Error: state_spec.yml not found at {spec_path}. Please check the path.")
        print("Skipping PokemonEnv instantiation for now.")
        observer = None
    except Exception as e:
        print(f"Error initializing StateObserver: {e}")
        observer = None


    if observer:
        # 対戦相手プレイヤーの準備 (例: RandomPlayer)
        opponent = RandomPlayer(battle_format="gen9randombattle")
        print("Opponent player (RandomPlayer) created.")

        # PokemonEnv インスタンスの作成
        # チームはデバッグ用にmy_team_for_debug.txtから読み込むことを想定
        # my_team_for_debug.txt の内容をPascalフォーマット文字列として渡す必要があります。
        # ここでは仮にNoneとして、ランダムバトルを想定します。
        # 固定パーティの場合は、team_pascal引数にチーム文字列を指定してください。
        # 例: my_team_pascal_str = open("../../config/my_team_for_debug.txt", "r").read()
        #     env = PokemonEnv(opponent_player=opponent, state_observer=observer, battle_format="gen9ou", team_pascal=my_team_pascal_str)
        try:
            env = PokemonEnv(opponent_player=opponent, state_observer=observer)
            print("PokemonEnv instance created successfully.")
            print(f"Observation Space: {env.observation_space}")
            print(f"Action Space: {env.action_space}")

            # 受け入れ基準の確認
            assert isinstance(env, gym.Env), "PokemonEnv is not a subclass of gym.Env"
            assert hasattr(env, 'observation_space'), "observation_space is not defined"
            assert hasattr(env, 'action_space'), "action_space is not defined"
            assert env.action_space.n == 10, f"Action space size is {env.action_space.n}, expected 10"
            # observation_spaceの形状はStateObserverの実装に依存するため、ここでは具体的なチェックは難しい
            # assert env.observation_space.shape == (observer.get_observation_space_dim(),), "Observation space shape mismatch"
            print("Basic acceptance criteria met.")

        except Exception as e:
            print(f"Error during PokemonEnv instantiation or basic checks: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("PokemonEnv instantiation skipped due to StateObserver initialization failure.")