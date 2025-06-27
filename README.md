# Maple

PokemonEnv を利用したポケモンバトル強化学習フレームワークです。

## 変更履歴

- 2025-06-13 `PokemonEnv.reset()` と `step()` に `return_masks` オプションを追加し、
  観測とあわせて利用可能な行動マスクを返すよう更新
- 2025-06-27 `EnvPlayer` が `battle.last_request` の変化を監視し、更新を確認してから
  `PokemonEnv` に `battle` オブジェクトを送信するよう改善
