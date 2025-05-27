# src/environments/pokemon_env.py

from websockets.exceptions import ConnectionClosedOK # ★ PokemonEnv.close() のために追加 ★
import asyncio
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
import logging
from typing import Optional, Tuple, Dict, Any


from poke_env.player import Player, RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
# from poke_env.player.baselines import MaxBasePowerPlayer # 現状未使用ならコメントアウト可
from poke_env.exceptions import ShowdownException


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.state.state_observer import StateObserver
# action_helper_module は __init__ の引数で渡されるので、ここでのインポートは不要
# import src.action.action_helper as action_helper_module

class PokemonEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, opponent_player: Player, state_observer: StateObserver, action_helper,
                 battle_format: str = "gen9ou", team_pascal: str = None, player_username: str = "MapleEnvPlayer"):
        super().__init__()

        self.battle_format = battle_format
        self.opponent = opponent_player
        self.state_observer = state_observer
        self.action_helper = action_helper
        self.current_battle: Optional[Battle] = None
        self._player_username = player_username
        self._player_password = None
        

        # --- 状態空間 (Observation Space) の定義 ---
        try:
            obs_dim = self.state_observer.get_observation_dimension()
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) # ★先に設定★
            print(f"Observation space set with dimension: {obs_dim}")
        except Exception as e:
            print(f"Error setting up observation space: {e}")
            raise e

        # --- 行動空間 (Action Space) の定義 ---
        self.action_space_size = 10 # 固定長10と仮定
        self.action_space = spaces.Discrete(self.action_space_size) # ★先に設定★

        env_player_account_config = AccountConfiguration(self._player_username, self._player_password)

        self.player = EnvPlayer(
            account_configuration=env_player_account_config,
            battle_format=self.battle_format,
            team=team_pascal,
            log_level=logging.DEBUG,
            env_ref=self
        )

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self._battle_is_over = True
        self._current_observation: Optional[np.ndarray] = None

        print("PokemonEnv initialized.")
        print(f"Observation Space: {self.observation_space}") # ★設定後にprint★
        print(f"Action Space: {self.action_space}")       # ★設定後にprint★

    async def _handle_battle_step(self, action_index: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        非同期で1ステップの対戦処理を行う内部メソッド
        """
        if self.current_battle is None or self._battle_is_over:
            print("Error: Battle not started or already over. Call reset() first.")
            dummy_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            if self._current_observation is not None:
                dummy_obs = self._current_observation
            # current_battle が None の場合、state_observer.observe に渡せないため修正
            # elif self.current_battle:
            #     dummy_obs = self.state_observer.observe(self.current_battle)
            return dummy_obs, 0.0, True, False, {"error": "Battle not started or over"}

        # 1. 行動をOrderに変換
        try:
            order = self.action_helper.action_index_to_order(self.player, self.current_battle, action_index)
            print(f"[{self.player.username}] Action index: {action_index} -> Order: {order}")
        except ValueError as e:
            print(f"Error converting action_index {action_index} to order: {e}")
            obs = self.state_observer.observe(self.current_battle)
            self._current_observation = obs
            # battle.finished を参照する前に battle オブジェクトの存在を確認
            terminated = self.current_battle.finished if self.current_battle else True
            return obs, -0.01, terminated, False, {"error": f"Invalid action {action_index}", "message": str(e)}

        self.player.set_next_action_for_battle(self.current_battle, order)

        try:
            await asyncio.wait_for(self.player.wait_for_battle_update(self.current_battle), timeout=10.0)
        except asyncio.TimeoutError:
            print(f"Warning: Timeout waiting for battle update in step for battle {self.current_battle.battle_tag if self.current_battle else 'N/A'}")
        except ShowdownException as e:
            print(f"ShowdownException during step: {e}")
            self._battle_is_over = True
            obs = self.state_observer.observe(self.current_battle) if self.current_battle else np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            self._current_observation = obs
            return obs, -1.0, True, True, {"error": "ShowdownException", "message": str(e)}

        if self.current_battle is None or not hasattr(self.current_battle, 'battle_tag'):
            print(f"Error: current_battle is None or invalid after step logic. Battle might have ended abruptly.")
            self._battle_is_over = True
            obs = self._current_observation if self._current_observation is not None else \
                  np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
            return obs, -1.0, True, False, {"error": "Battle object became invalid"}

        observation = self.state_observer.observe(self.current_battle)
        self._current_observation = observation

        reward = 0.0
        terminated = self.current_battle.finished
        truncated = False

        if terminated:
            self._battle_is_over = True
            if self.current_battle.won:
                reward = 1.0
            # battle.lost は Player クラスのプロパティなので、 AbstractBattle にはない
            # battle.won が False で finished なら負けか引き分けと判断
            elif not self.current_battle.won : # 勝利していない場合
                reward = -1.0 # 敗北時の報酬
            # else: # 引き分け (wonでもなくlostでもない場合だが、poke-envではwon=Falseで敗北も含む)
            # 上記のelifで敗北はカバーされるので、厳密な引き分けの判定はwonでもなく、サーバーメッセージで'tie'となる場合
            # ここではシンプルに勝利以外を-1にしている。引き分けを0にするなら別途条件分岐が必要。
            # 例えば、battle.tied (poke-env v0.6.5以降でBattleクラスにあり) などで判定
            if hasattr(self.current_battle, 'tied') and self.current_battle.tied:
                 reward = 0.0 # 引き分けの場合
            print(f"Battle finished. Won: {self.current_battle.won}, Tied: {hasattr(self.current_battle, 'tied') and self.current_battle.tied}, Reward: {reward}")


        info = {
            "action_index": action_index,
            "order_str": order_str if 'order_str' in locals() else "N/A",
            "turn": self.current_battle.turn,
            "finished": terminated,
            "won": self.current_battle.won if terminated else None,
            "my_active_hp_frac": self.current_battle.active_pokemon.current_hp_fraction if self.current_battle.active_pokemon else 0,
            "opponent_active_hp_frac": self.current_battle.opponent_active_pokemon.current_hp_fraction if self.current_battle.opponent_active_pokemon else 0,
        }
        if hasattr(self.current_battle, 'trapped') and self.current_battle.trapped: # AbstractBattleにはtrappedがない場合がある
            info["trapped"] = self.current_battle.trapped

        return observation, reward, terminated, truncated, info

    async def _reset_env(self, seed=None, options=None):
        """
        非同期で環境をリセットして初期観測を返します。
        Gymnasium の reset() から呼ばれる内部コルーチン。

        Returns
        -------
        observation : np.ndarray
            初期状態ベクトル
        info : dict
            デバッグ用の追加情報
        """

        # 0) 乱数シード（必要ならここで使う）
        if seed is not None:
            np.random.seed(seed)



        # 2) プレイヤー側 WebSocket を起動（まだなら）
        if not self.player.ps_client.is_listening:
            print(f"[DBG] Starting player WebSocket listener for {self.player.username}")
            # listen() は無限ループなので Task として fire-and-forget
            listen_task = self.loop.create_task(self.player.ps_client.listen())
            self.player.ps_client._listening_coroutine = listen_task

        # opponent は __init__ で start_listening=True なので通常は動いているが、
        # 念のためチェック
        if not self.opponent.ps_client.is_listening:
            self.loop.create_task(self.opponent.ps_client.listen())

        # 3) ログイン完了を待つ
        await asyncio.gather(
            self.player.ps_client.wait_for_login(),
            self.opponent.ps_client.wait_for_login(),
        )

        # 4) プレイヤ内部のバトルトラッカーをリセット
        self.player.reset_battles()
        self.opponent.reset_battles()
        self.current_battle = None
        self._battle_is_over = False

        # 5) チャレンジを送ってバトルを開始
        #    Player.send_challenges / accept_challenges はバトル終了までブロック
        #    するのでバックグラウンドタスクとして動かす
        challenge_task = self.loop.create_task(
            self.player.send_challenges(self.opponent.username, 1)
        )
        accept_task = self.loop.create_task(
            self.opponent.accept_challenges(self.player.username, 1)
        )

        # 6) self.current_battle がセットされるまで待機
        timeout = 30.0  # 秒
        start_t = self.loop.time()
        while self.current_battle is None:
            if self.loop.time() - start_t > timeout:
                raise TimeoutError(
                    "Battle did not start within %.1f seconds." % timeout
                )
            await asyncio.sleep(0.1)

        # choose_move() が呼ばれて current_battle が出来た時点で
        # active_pokemon 等の request も揃っているはずだが、
        # available_moves が空の可能性もあるため念のためリトライ
        while (
            self.current_battle.active_pokemon is None
            or len(self.current_battle.available_moves) == 0
        ):
            await asyncio.sleep(0.05)

        # 7) 観測値を生成
        observation = self.state_observer.observe(self.current_battle)
        self._current_observation = observation

        info = {
            "battle_tag": self.current_battle.battle_tag,
            "turn": self.current_battle.turn,
            "opponent": self.opponent.username,
        }

        return observation, info

    async def smoke_test():
        obs, info = env.reset()
        print("First obs shape:", obs.shape, "info:", info)
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, truncated, inf = env.step(action)
            total_reward += reward
        print("Episode finished. Total reward:", total_reward)


    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # アクションインデックスをBattleOrderに変換
        order = self.action_helper.action_index_to_order(self.player, self.current_battle, action_index)
        # BattleOrderをEnvPlayerにセットして次の行動に指定
        self.player.set_next_action_for_battle(self.current_battle, order)
        # 非同期のバトル進行処理が完了するまで待機
        if self.loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(self._handle_battle_step(action_index), self.loop)
            return fut.result()
        else:
            return self.loop.run_until_complete(self._handle_battle_step(action_index))

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Gymnasium 互換 reset().
        非同期処理を包むだけで、戻り値は (observation, info)。
        """
        if self.loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(
                self._reset_env(seed, options), self.loop
            )
            return fut.result()
        else:
            return self.loop.run_until_complete(
                self._reset_env(seed, options)
            )

    def render(self, mode='human'):
        # renderメソッドの実装は変更なしでOK
        if mode == 'human':
            if self.current_battle and not self._battle_is_over:
                print(f"\n--- Turn: {self.current_battle.turn} ---")
                my_active = self.current_battle.active_pokemon
                opp_active = self.current_battle.opponent_active_pokemon
                if my_active:
                    moves_info = []
                    for move_idx, move_obj in enumerate(my_active.moves.values()): # my_active.moves.values() を使用
                         moves_info.append(f"{move_obj.id}({move_obj.current_pp}/{move_obj.max_pp})")
                    print(f"My Active: {my_active.species} (HP: {my_active.current_hp_fraction*100:.1f}%) Status: {my_active.status} Moves: {moves_info}")
                if opp_active:
                    print(f"Opponent Active: {opp_active.species} (HP: {opp_active.current_hp_fraction*100:.1f}%) Status: {opp_active.status}")

                if hasattr(self.action_helper, 'get_available_actions_with_details'):
                    _, detailed_actions = self.action_helper.get_available_actions_with_details(self.current_battle)
                    print("Available Actions:")
                    for idx, action_info in detailed_actions.items():
                        print(f"  {idx}: {action_info['type']} - {action_info['name']}")
                elif hasattr(self.action_helper, 'get_available_actions'):
                    action_mask, action_mapping = self.action_helper.get_available_actions(self.current_battle)
                    print("Available Actions (Index: Type, Sub-Index):")
                    for i in range(len(action_mask)):
                        if action_mask[i] == 1:
                             print(f"  {i}: {action_mapping.get(i)}")
            elif self._battle_is_over:
                print("Battle is over. Call reset() to start a new one.")
            else:
                print("Battle not started yet for render. Call reset() first.")

    

    async def close(self):
        print(f"Closing PokemonEnv (Player: {self.player.username}).")
        tasks_to_await = []
        results = await asyncio.gather(*tasks_to_await, return_exceptions=True)
        for task_result, who in zip(results, ["EnvPlayer", "Opponent"]):
            if isinstance(task_result, Exception) and not isinstance(task_result, ConnectionClosedOK):
                print(f"[close] {who}: {type(task_result).__name__} - {task_result}")

        
                    
        # EnvPlayer のクリーンアップ
        # EnvPlayer 側は listen していない可能性があるのでガード
        if self.player.ps_client.is_listening:          # ← 追加
            tasks_to_await.append(self.player.ps_client.stop_listening())
        if self.opponent.ps_client.is_listening:
            tasks_to_await.append(self.opponent.ps_client.stop_listening())
            
        if tasks_to_await:
            try:
                results = await asyncio.gather(*tasks_to_await, return_exceptions=True)
                for i, result in enumerate(results):
                    player_name = "EnvPlayer" if i == 0 and self.player else "Opponent"
                    if isinstance(result, Exception):
                        if not isinstance(result, ConnectionClosedOK):
                            print(f"Exception during stop_listening for {player_name}: {type(result).__name__} - {result}")
                    else:
                        print(f"stop_listening for {player_name} completed.")
                print("Async cleanup tasks in close() processed.")
            except Exception as e:
                print(f"Error during asyncio.gather in close: {e}")

        print("PokemonEnv close() method finished.")
class EnvPlayer(Player):
    def __init__(self, account_configuration: AccountConfiguration, battle_format: str, team: str = None, log_level: int = None, env_ref: Optional[PokemonEnv] = None):
        super().__init__(
            account_configuration=account_configuration,
            battle_format=battle_format,
            team=team,
            log_level=log_level,
            start_listening=False # PokemonEnvのresetで開始
        )
        # self._opponent_ai: Optional[Player] = None # EnvPlayerは相手AIを直接持たない
        self._current_battle_for_player: Optional[Battle] = None
        self._env_ref = env_ref

        self._choose_move_called_event = asyncio.Event()
        self._action_to_send: Optional[str] = None

        self._battle_update_event = asyncio.Event()
        
        # poke-env が使うコールバックを差し替え
        self.ps_client._handle_battle_message = self._handle_battle_message



    def choose_move(self, battle: Battle):
        # Envからの次アクションがセット済みの場合はそれを取得（BattleOrder型）
        # 初回呼び出し時に現在のBattleを環境に登録
        if self._env_ref is not None and self._env_ref.current_battle is None:
            self._env_ref.current_battle = battle
            self._current_battle_for_player = battle
                    
        if self._action_to_send:
            action_order = self._action_to_send  # 例: BattleOrderオブジェクト
            self._action_to_send = None
            return action_order  # BattleOrderなのでmessage属性あり
        else:
            # 未設定ならFutureを作成し、Env側のstep()からの入力を待つ
            loop = asyncio.get_event_loop()
            self._next_action_future = loop.create_future()
            return self._next_action_future

    def set_next_action_for_battle(self, battle: Battle, action_order: Optional [BattleOrder]): 
        # （バトルIDのチェックはそのまま）
        if self._current_battle_for_player and self._current_battle_for_player.battle_tag != battle.battle_tag:
            print(f"Warning [{self.username}]: set_next_action called for battle {battle.battle_tag}, but current is {self._current_battle_for_player.battle_tag}")
        # Futureが未完了で待機中なら結果をセットして解決する
        if hasattr(self, "_next_action_future") and not self._next_action_future.done():
            self._next_action_future.set_result(action_order)    
        else:
            # 念のためFallback（通常はこちらは通らない想定）
            self._action_to_send = action_order            
                
        self._action_to_send: Optional[BattleOrder] = action_order # order文字列を保存
        self._choose_move_called_event.set()

    async def _handle_battle_message(self, split_messages: list[list[str]]):
        """poke-env から受け取った 1 バッチ分のメッセージを処理する。

        `split_messages` は「ルームヘッダ行」と複数の
        `|XXX|...` 行が 1 つのリストにまとまった構造になる。
        強制交代時は `|request|…` 1 行だけ、というケースがあるので
        **先頭も含めて** 全メッセージを調べる。
        """
        await super()._handle_battle_message(split_messages)
        print(f"リクエスト検知 -------------------:{split_messages}")

        trigger_tags = {"request", "win", "lose", "tie", "error"}
        if any(len(m) > 1 and m[1] in trigger_tags for m in split_messages):
            # 新しい request / ターン進行 / バトル終了 など
            # いずれかを検知したら待機中コルーチンを起こす
            self._battle_update_event.set()

    async def _battle_message(self, battle: Battle, message: str):
        # ❶ まず battle を更新
        await super()._battle_message(battle, message)
        # ❷ その後にイベントを通知
        if message.startswith(('|turn|', '|win|', '|lose|', '|tie|', '|error|', '|upkeep|')):
            self._battle_update_event.set()
            
    async def _battle_finished_callback(self, battle: Battle):
        # バトル終了フラグをEnv側に通知
        if hasattr(self, '_env_instance') and self._env_instance:
            self._env_instance.current_battle_finished = True
        # 親クラスのコールバックを同期的に呼ぶ（戻り値なし）
        super()._battle_finished_callback(battle)  # 非同期でないためawaitしない

    async def wait_for_battle_update(self, battle: Battle):
        prev_turn  = battle.turn
        prev_rqid = battle.last_request.get("rqid", -1) if hasattr(battle, "last_request") else -1
        print(f"[DBG] wait_for_battle_update enter "
              f"prev_turn={prev_turn}, prev_rqid={prev_rqid}")
        while True:
            await self._battle_update_event.wait()
            self._battle_update_event.clear()
            cur_turn = battle.turn
            cur_rqid = battle.last_request.get("rqid", -1) if hasattr(battle, "last_request") else -1
            print(f"[DBG] wake: turn={cur_turn}, rqid={cur_rqid}, "
                  f"wait={getattr(battle,'_wait',None)}, finished={battle.finished}")

            # ターン番号か rqid が進んだか　バトルが終了したタイミングで抜ける
            # サーバから行動受付中 (battle._wait == False) になった瞬間だけ抜ける
            if (
                (cur_rqid > prev_rqid or cur_turn > prev_turn)  # 何か新しい request が来た
                and not battle._wait                          # しかも wait フラグが外れた
            ) or battle.finished:
                break


# --- 動作確認のための仮コード ---
async def main_test():
    print("PokemonEnv class defined. Basic structure is ready.")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    try:
        spec_path = os.path.join(project_root, "config/state_spec.yml")
        observer = StateObserver(spec_path)
        print(f"StateObserver loaded with spec: {spec_path}")
        print(f"Calculated observation dimension: {observer.get_observation_dimension()}")
    except Exception as e:
        print(f"Error initializing StateObserver or getting dimension: {e}")
        import traceback
        traceback.print_exc()
        return

    opponent_account_config = AccountConfiguration("OpponentRandomTest", None) # ユーザー名を変更して衝突を避ける
    opponent_player = RandomPlayer(
        account_configuration=opponent_account_config,
        battle_format="gen9ou",
        log_level=logging.INFO,
        start_listening=True # RandomPlayer は listen 状態で待機させる
    )
    print("Opponent player created and starts listening.")

    try:
        team_str = None
        team_file_path = os.path.join(project_root, "config/my_team_for_debug.txt")
        try:
            with open(team_file_path, "r") as f:
                team_str = f.read()
            print(f"Team loaded from {team_file_path}")
        except FileNotFoundError:
            print(f"Warning: {team_file_path} not found.")

        # action_helper モジュールをインポート
        from src.action import action_helper as action_helper_module

        env = PokemonEnv(
            opponent_player=opponent_player,
            state_observer=observer,
            action_helper=action_helper_module, # ★action_helperモジュールを渡す★
            battle_format="gen9ou",
            team_pascal=team_str,
            player_username="MapleEnvPlayerTest"
        )
        print("PokemonEnv instance created successfully.")
        # print(f"Observation Space: {env.observation_space}") # __init__内でprint済
        # print(f"Action Space: {env.action_space}")       # __init__内でprint済

        assert isinstance(env, gym.Env)
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert env.action_space.n == 10
        assert env.observation_space.shape == (observer.get_observation_dimension(),)
        print("Basic acceptance criteria met.")

    except Exception as e:
        print(f"Error during PokemonEnv instantiation or basic checks: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ─────────────────────────────────────────────
        # ここを env.close() だけに統一
        # ─────────────────────────────────────────────
        if 'env' in locals() and env:
            # 非同期で定義した close() を呼び出すには await が必須
            print(f"Closing PokemonEnv ({env.player.username}) in main_test finally...")
            try:
                await env.close()        # ← opponent も内部で stop_listening される
            except Exception as e:
                print(f"Error during env.close() in main_test finally: {type(e).__name__} - {e}")


if __name__ == '__main__':
    asyncio.run(main_test())