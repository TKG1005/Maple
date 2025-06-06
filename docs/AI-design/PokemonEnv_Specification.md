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
| メッセージフロー | 1. サーバーが `"teamPreview": true` を含む `request` を送り選出ポケモンを要求<br>2. 両プレイヤーが `/team` 送信(現実装では先頭 3 匹を選択)<br>3. 以降各ターンで `request` が届き `/choose …` を返信<br>4. サーバーが結果をブロードキャスト |

* 各 `request` には昇順の `rqid` が付与され、乱序で届くことがある
* `forceSwitch` が `True` の場合、同一ターンに複数の `request` が送られる

---

## 3. 同期 / 非同期処理

* 外部: `poke_env` は **asyncio**‐ベースで WebSocket を管理  
* PokemonEnv API: **同期的** (`reset()`, `step()`)  
* 手順  
  1. `reset()` で `play_against` を呼び、裏で非同期タスクが起動  
  2. `step(action)` は行動インデックスを **BattleOrder** に変換し送信  
  3. poke-envはPSClient.listen()でshowdownサーバーからのメッセージを待機
  4. poke-envは `request` を含むJSONを受信すると Battle.parse_request()を実行して、`poke_env` 内の `Battle` オブジェクトが更新される
  5. `PokemonEnv` はpoke-envがshowdownサーバから受け取るJSONを監視して、`request` を含むメッセージを検知して`Battle` オブジェクトが更新されるのを待つ
  6. `PokemonEnv` はメッセージの内容に応じてエージェントの対応するメソッドを呼ぶ("teamPreview":trueならchoose_team,その他の場合はchoose_move())
* 注意
* `request` を含むメッセージは必ずしも順番通りには届かない(rqid=n+1のメッセージがrqid=nのメッセージの後に届く場合がある)ので最新のrqidに反応する必要がある
* 1ターンに複数の`request` を含むメッセージが来ることがある(交代選択が必要な場合:(forceSwitch=True))
* `step()` 実行後は最新 `rqid` の `request` を処理し `battle.turn` が増加するまで待機する
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
    participant Agent
    participant PokemonEnv
    participant poke_env/EnvPlayer
    participant Showdown
    Agent->>PokemonEnv: reset()
    PokemonEnv->>EnvPlayer: play_against()
    EnvPlayer->>Showdown: /team
    Showdown-->>EnvPlayer: request(teamPreview=true)
    PokemonEnv->>EnvPlayer: select_team()
    EnvPlayer->>Showdown: /choose team (1,2,3)
    Showdown-->>EnvPlayer: state
    EnvPlayer-->>PokemonEnv: Battle
    note over PokemonEnv: 観測ベクトル生成(state_observer.pyを使用)
    PokemonEnv->>Agent: 観測ベクトル
    Agent: 行動空間ベクトルを取得(action_helper.pyを使用), アルゴリズムに基づいて行動を決定(現時点ではランダム選択)
    Agent->>PokemonEnv: step(action_idx)
    PokemonEnv->>EnvPlayer: choose_move(BattleOrder)
    EnvPlayer->>Showdown: /choose …
    Showdown-->>EnvPlayer: state
    EnvPlayer-->>PokemonEnv: Battle
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
* **メッセージ監視**: `PokemonEnv` は `poke_env` が受信した `request` を監視し、
  `teamPreview` を含む場合は `select_team()`、それ以外は `choose_move()` を呼び出す
* **チームプレビュー**: 対戦開始時に Showdown サーバーから `"teamPreview": true` を含む `request` JSON が届いたら、PokemonEnv はエージェントのポケモン選択メソッド `select_team()` を呼び出し `/choose team` を送信する。デフォルト実装では登録順先頭 3 匹を選出
* **再利用接続**: 各エピソード開始時に `reset_battles()`
* **step 待機処理**: 最新 `rqid` の `request` を処理して `battle.turn` が進むまでループ
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
