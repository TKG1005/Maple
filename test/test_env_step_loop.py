"""
Task 1.4 ― PokemonEnv のスモークテスト
-------------------------------------
・PokemonEnv を 3 エピソード走らせて，各ターン正常に
  obs/reward/terminated が返ってくるかを確認します。
・クリーンアップも含め，全処理は blocking（同期的）に書いています。
"""

import os
import sys
import logging
import asyncio
import numpy as np
from poke_env.player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.environment.battle import Battle

# Maple パッケージのインポート
# ***** Maple プロジェクト内モジュール *****
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.environments.pokemon_env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper   # 必ずモジュールを渡す

# ---------- 前提ファイルの場所 ----------

SPEC_PATH    = os.path.join(PROJECT_ROOT, "config", "state_spec.yml")
TEAM_PATH    = os.path.join(PROJECT_ROOT, "config", "my_team.txt")


# -----------------------------

def run_episode(env, episode_num: int) -> float:
    """
    1 エピソードだけ実行し，得られた総報酬を返す。

    Gymnasium では step() / reset() は同期メソッドだが，
    内部で asyncio を使っているため裏側でイベントループが動いている。
    ここでは意識せずシンプルに呼び出せば OK。
    """
    # reset() で最初の観測と追加情報を取得
    obs, info = env.reset()
    print(f"\n=== Episode {episode_num+1} started ===")
    print(f"Initial obs dim = {obs.shape}, info = {info}")

    done   = False        # terminated フラグ
    cum_r  = 0.0          # 累積報酬
    t      = 0            # ターン数

    # 終了フラグが立つまでループ
    while not done:
        t += 1
        mask, _ = action_helper.get_available_actions(env.current_battle)
        print(f"[DBG]action_mask = {mask}")
        valid_indices = np.where(mask == 1)[0]
        action = int(np.random.choice(valid_indices))  #利用可能な行動からランダムに１つを選ぶ
        print(f"選択した行動 {action}")

        next_obs, reward, done, truncated, inf = env.step(action)
        cum_r += reward

        # --- 学習用ログ（step 毎） ---
        print(f"[T{t:02d}] act={action}, "
              f"rew={reward:+.1f}, done={done}, ")


    print(f"Episode {episode_num+1} finished in {t} turns. "
          f"total_reward = {cum_r:+.1f}")
    return cum_r


def main():
    # ① StateObserver を作る
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    spec_path = os.path.join(project_root, "config", "state_spec.yml")
    observer = StateObserver(spec_path)

    with open(TEAM_PATH, encoding="utf-8") as f:
        team_str = f.read()
        
    # ② 対戦相手（完全ランダム）
    opp_cfg = AccountConfiguration("OpponentRandomTest", None)
    opponent = RandomPlayer(account_configuration=opp_cfg,
                            battle_format="gen9ou",
                            log_level=logging.DEBUG,
                            team=team_str,  # ★ここを追加
                            start_listening=True)

    # ③ PokemonEnv を初期化
    env = PokemonEnv(opponent_player=opponent,
                     state_observer=observer,
                     action_helper=action_helper,
                     battle_format="gen9ou",
                     team_pascal=team_str,
                     player_username="MapleEnvPlayer")

    # ④ テスト実行
    NUM_EPISODES = 3
    rewards = []
    for ep in range(NUM_EPISODES):
        rewards.append(run_episode(env, ep))

    print("\n=== Summary ===")
    for i, r in enumerate(rewards):
        print(f" Episode {i+1}: {r:+.1f}")
    print("===============")

    # ⑤ 全 WebSocket をクローズ
    env.close()


if __name__ == "__main__":
    main()
    # asyncio の残タスクが確実に終了する猶予を与える
    asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))
