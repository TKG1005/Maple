1. `src/rewards/` ディレクトリに新しい報酬クラス `FailAndImmuneReward` を作成する。

   * `RewardBase` を継承する。
   * `reset()` と `calc(battle)` メソッドを実装する。
   * `calc` は `battle.last_invalid_action` が `True` の場合、`-0.02` を返す。

2. 報酬クラスを登録する。

   * `src/rewards/__init__.py` でエクスポートする。
   * `CompositeReward.DEFAULT_REWARDS` に `"fail_immune": FailAndImmuneReward` を追加する。

3. `copy_of_poke-env/poke_env/environment/abstract_battle.py` を編集する。

   * `MESSAGES_TO_IGNORE` から `"-fail"` と `"immune"` を削除する。
   * `parse_message` で `"-fail"` および `"-immune"` イベントを処理する処理を追加する。

     * 影響を受けたポケモンが `self._player_role` 側なら、`self._last_invalid_action = True` とする。

4. `Battle` クラスに `_last_invalid_action` 属性とプロパティを追加する。

   * `Battle.__init__` で `False` に初期化する。
   * `last_invalid_action` のgetterと `reset_invalid_action()` メソッドを追加する。

5. `PokemonEnv.step` で各バトルごとにこのフラグを報酬計算前にリセットし、`True` になったときはコンソールにメッセージを表示する。

6. `src/env/pokemon_env.py` を更新し、`CompositeReward` 使用時に新しい報酬が加わるよう `_calc_reward` を修正する。

7. ユニットテストを拡張する。

   * `FailAndImmuneReward` が `battle.last_invalid_action` がセットされた時にペナルティを課すことを検証する。
   * 複合報酬（CompositeReward）のテストも新しい報酬マッピングを含むよう調整する。
