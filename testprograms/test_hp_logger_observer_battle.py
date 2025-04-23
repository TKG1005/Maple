
import asyncio
import time
from my_simple_player import MySimplePlayer
from poke_env.player import RandomPlayer
from poke_env.environment.battle import Battle
from state_observer import StateObserver
with open("my_team.txt", encoding="utf-8") as f:
    team_str = f.read()
    

class HPLoggingObserverPlayer(MySimplePlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer = StateObserver("state_spec.yml")  # YAMLファイルのパス
        
        # ① Team Preview 専用メソッドを実装
    def teampreview(self, battle):
        print('チームプレビューを行います')
        # 先頭（index 0）をリードに選ぶだけの仮実装
        return "/team 123456"  # 6→3 ならどの順でも OK

    # ② 通常ターンの choose_move

    def choose_move(self, battle: Battle):    
        # Team Preview 中は turn==0 なのでベクトル生成をスキップ
        if battle.turn == 0:
            return self.choose_default_move()

        state_vec = self.observer.observe(battle)
        # 状態ベクトルから自分と相手のHP割合を抜き出す（添字は仮）
        my_hp = battle.active_pokemon.current_hp_fraction
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction
        print(f"[Turn {battle.turn}] 自分HP: {my_hp:.2f} / 相手HP: {opp_hp:.2f}")
        print("状態ベクトルの長さ:", len(state_vec))
        return super().choose_move(battle)

async def main():

    player1 = HPLoggingObserverPlayer(battle_format="gen9randombattle",log_level=25)
    player2 = RandomPlayer(battle_format="gen9randombattle", log_level=25)

    print("テスト対戦を開始します（StateObserver使用）...")
    await player1.battle_against(player2, n_battles=1)
    print("対戦終了。")

    if player1.battles:
        battle_result = list(player1.battles.values())[-1]
        final_turn = battle_result.turn
        if battle_result.won:
            print(f"勝者: HPLoggingObserverPlayer / 最終ターン: {final_turn}")
        elif battle_result.lost:
            print(f"勝者: RandomPlayer / 最終ターン: {final_turn}")
        elif battle_result.tied:
            print(f"結果: 引き分け / 最終ターン: {final_turn}")
        else:
            print(f"結果: 不明 / 最終ターン: {final_turn}")
    else:
        print("結果: 対戦結果を取得できませんでした。")

if __name__ == "__main__":
    asyncio.run(main())
