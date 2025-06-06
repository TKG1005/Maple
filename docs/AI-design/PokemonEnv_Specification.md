# PokemonEnv 技術仕様書 — Maple Project

## 1. 目的
本ドキュメントは、Maple プロジェクトの `pokemon_env.py` に定義された **PokemonEnv** クラスの技術仕様を、LLM がコード生成・補完の際に参照できる形式でまとめたものです。

---

## 2. poke_env と Showdown サーバー間の通信

| 項目 | 概要 |
| --- | --- |
| 接続方式 | **WebSocket** (`ws://localhost:8000`) |
| 使用ライブラリ | `poke_env` の `Player` と `ServerConfiguration` |
| 通信プロトコル | Pokémon Showdown テキストコマンド (`/team`, `/choose move 1`, など) |
| 対戦開始 | `EnvPlayer.play_against(opponent, n_battles=1)` |
| メッセージフロー | 1. サーバーが `"teamPreview": true` を含む `request` を送り選出ポケモンを要求<br>2. 両プレイヤーが `/team` 送信(ランダムで3匹を選択)<br>3. 以降各ターンで `request` が届き `/choose …` を返信<br>4. サーバーが結果をブロードキャスト |

* 各 `request` には昇順の `rqid` が付与され、乱序で届くことがある
* `forceSwitch` が `True` の場合、同一ターンに複数の `request` が送られる

---

## 3. 同期 / 非同期処理

* 外部: `poke_env` は **asyncio**‐ベースで WebSocket を管理  
* PokemonEnv API: **同期的** (`reset()`, `step()`)  

* 手順
  1. `reset()` で `play_against()` を呼び、裏で非同期タスクが起動
  2. `step(action)` は行動インデックスを `_action_queue` に投入
  3. Showdown から `request` が届くと poke-env が `EnvPlayer.teampreview()` または `EnvPlayer.choose_move()` を呼び出す
  4. `EnvPlayer.choose_move()` はキューから `action_idx` を受け取り `BattleOrder` を生成して返す
  5. `poke-env` が `/choose` を送信し `Battle` オブジェクトを更新
  6. `step()` は `battle.turn` が進むまで待機し、観測と報酬を返す

* 注意
* `request` は順不同で届くことがあるが、`Battle` オブジェクトが常に最新状態を保持するため、キュー投入済みの行動をそのまま処理できる
* 交代要求など複数の `request` が続くケースも、`choose_move()` が逐次呼び出されることで対処できる
* `step()` は `battle.turn` が変化しない場合に備えてタイムアウトを設ける
---

## 4. 観測（状態）空間

* 型: `gymnasium.spaces.Box(low=0, high=1, shape=(N,), dtype=np.float32)`
* 生成: `StateObserver`  
* 特徴量例  
  * 自分 / 相手アクティブポケモンの HP%, 種族, タイプ一 hot, 状態異常  
  * ベンチ 1・2 の HP%, 存在フラグ  
  * 技 1–4 の威力, タイプ一 hot, PP%  
  * 場の天候, フィールド, ターン数  
* One‑Hot 化・線形スケーリングで 0‑1 に正規化  
* 次元数: `StateObserver.get_observation_dimension()` で算出  

```text
observation = concat(
    own_active_stats,
    own_bench_1, own_bench_2,
    opp_active_stats,
    ...,
    global_field_info
)
```

---

## 5. 行動空間

```
gymnasium.spaces.Discrete(10)  # index 0‑9
```

| Index | 意味 | 備考 |
| --- | --- | --- |
| 0‑3 | 技 1‑4 | `create_order(move_i)` |
| 4‑7 | テラスタルして技 1‑4 | `create_order(move_i, terastallize=True)` |
| 8‑9 | ベンチ 1 or 2 に交代 | `create_order(pokemon_j)` |

* **ActionHelper**  
  * `get_available_actions(battle) -> mask[10], mapping`  
  * `action_index_to_order(player, battle, idx) -> BattleOrder`  

---

## 6. `reset()` / `step()` フロー

```mermaid
sequenceDiagram
    participant Agent(EnvPlayer)
    participant PokemonEnv
    participant EnvPlayer
    participant Showdown
    Agent->>PokemonEnv: reset()
    PokemonEnv->>EnvPlayer: play_against()
    Showdown-->>EnvPlayer: request("teamPreview":true)
    EnvPlayer->>Showdown: /team 123(現時点では先頭3匹を返す)
    Agent->>PokemonEnv: step(action_idx)
    PokemonEnv->>EnvPlayer: put(action_idx)
    Showdown-->>EnvPlayer: request
    EnvPlayer->>Showdown: /choose … (queueから取得)

    PokemonEnv-->>Agent: obs, reward, done
```

---

## 7. 報酬・エピソード終了

| 状況 | `terminated` | `reward` |
| --- | --- | --- |
| 自分が勝利 | True | +1 |
| 自分が敗北 | True | -1 |
| ターン > `MAX_TURNS` | True (`truncated`) | 0 |
| 途中ターン | False | 0 |

---

## 8. 実装ノート

* **遅延インポート**: `poke_env` は `reset()` 内でインポート
* **EnvPlayer**: 行動アルゴリズムは外部エージェントに委任
* **非同期キュー**: `PokemonEnv.step()` と `EnvPlayer.choose_move()` を接続するため `_action_queue` を利用
* **チームプレビュー**: `EnvPlayer.teampreview()` でチーム選択を行い `/choose team` を送信（デフォルトは先頭 3 匹を選出）
* **再利用接続**: 各エピソード開始時に `reset_battles()`
* **step 待機処理**: `battle.turn` が進むまで非同期でループし、タイムアウトを設ける
* **未実装**: `render()`, `close()` は将来拡張
* **依存**: `poke-env>=0.9`, Showdown server (localhost:8000)

---

## 9. 参考コードスニペット

```python
# 環境ベクトル取得
state: np.ndarray = state_observer.observe(battle)

# チームプレビュー処理
env._handle_team_preview(battle)

# 行動マスク取得
mask, mapping = action_helper.get_available_actions(battle)

# 行動インデックス -> BattleOrder
order = action_helper.action_index_to_order(env_player, battle, idx)
```

---

### End of File
