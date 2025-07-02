# Maple

PokemonEnv を利用したポケモンバトル強化学習フレームワークです。

## 変更履歴

- 2025-06-13 `PokemonEnv.reset()` と `step()` に `return_masks` オプションを追加し、
  観測とあわせて利用可能な行動マスクを返すよう更新
- 2025-06-27 `EnvPlayer` が `battle.last_request` の変化を監視し、更新を確認してから
  `PokemonEnv` に `battle` オブジェクトを送信するよう改善
- 2025-06-28 `train_selfplay.py` にチェックポイント保存オプションを追加し、一定間隔で
  モデルを自動保存可能に
- 2025-06-29 対戦ログ比較用の `plot_compare.py` を新規追加し、学習結果をグラフで確認できるように
- 2025-06-29 `SingleAgentCompatibilityWrapper` の `reset()` と `step()` が `return_masks` を受け取り
  `PokemonEnv` の行動マスクと連携するよう更新
- 2025-07-01 PPO 対応手順をまとめた `docs/M7_setup.md` を追加し、`train_selfplay.py` の `--algo` オプションでアルゴリズムを切り替え可能に
- 2025-07-05 `HPDeltaReward` を追加し、`--reward hp_delta` オプションで使用可能に
- 2025-07-06 `train_selfplay.py` の報酬を `CompositeReward` ベースに変更し、
  `config/train_config.yml` の設定で他の報酬へ切り替え可能に
