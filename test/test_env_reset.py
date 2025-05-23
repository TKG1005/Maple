"""
env.reset() のスモークテスト
----------------------------------
✓ バトルが開始される（battle_tag が埋まる）
✓ 観測値が numpy.ndarray
✓ info dict に最低限のキーが入っている
"""

import os
import asyncio
import numpy as np
import logging
import sys



from poke_env.player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration

# ***** Maple プロジェクト内モジュール *****
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.state.state_observer import StateObserver
from src.action import action_helper            # モジュールごと渡す実装に合わせる
from src.environments.pokemon_env import PokemonEnv

# ---------- 前提ファイルの場所 ----------

SPEC_PATH    = os.path.join(PROJECT_ROOT, "config", "state_spec.yml")
TEAM_PATH    = os.path.join(PROJECT_ROOT, "config", "my_team_for_debug.txt")

# ---------- テスト本体 ----------
async def _run():
    # ① StateObserver
    observer = StateObserver(SPEC_PATH)
    
    with open(TEAM_PATH, encoding="utf-8") as f:
        team_str = f.read()

    # ② 対戦相手（ランダム AI）
    opp_acct = AccountConfiguration("ResetTestOpponent", None)
    opponent = RandomPlayer(
        account_configuration=AccountConfiguration("OpponentRandomTest", None),
        battle_format="gen9ou",
        team=team_str,              # ★ここを追加
        log_level=logging.INFO,
        start_listening=True,
    )

    # ③ 自分の環境
    env = PokemonEnv(
        opponent_player=opponent,
        state_observer=observer,
        action_helper=action_helper,   # モジュールそのものを渡す
        battle_format="gen9ou",
        team_pascal=team_str,
        player_username="ResetTester",
    )

    # ④ reset() を呼び出す
    observation, info = await asyncio.to_thread(env.reset, seed=123)

    # ⑤ アサーション（失敗すれば例外で停止）
    assert isinstance(observation, np.ndarray), "observation は ndarray ではありません"
    assert info.get("battle_tag"), "info['battle_tag'] が空です"
    print("✓ env.reset() で ndarray が返り battle_tag が取得できました")
    print("  shape:", observation.shape, "| dtype:", observation.dtype)
    print("  info :", {k: info[k] for k in ('battle_tag','turn','opponent') if k in info})

    # ⑥ 後片付け
    await env.close()

# 直接実行したときだけ asyncio.run
if __name__ == "__main__":
    asyncio.run(_run())
