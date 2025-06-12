# PokemonEnv 技術仕様書 — Maple Project

## 1. 目的
本ドキュメントは、Maple プロジェクトの `pokemon_env.py` に定義された **PokemonEnv (Multi‑Agent Edition)** の技術仕様を、LLM がコード生成・補完の際に参照できる形式でまとめたものです。

---

## 2. poke_env と Showdown サーバー間の通信

| 項目 | 概要 |
| --- | --- |
| 接続方式 | **WebSocket** (`ws://localhost:8000`) |
| 使用ライブラリ | `poke_env` の `Player` と `ServerConfiguration` |
| 通信プロトコル | Pokémon Showdown テキストコマンド (`/team`, `/choose move 1`, など) |
| 対戦開始 | `EnvPlayer.play_against(opponent, n_battles=1)`を2インスタンス同時に起動し、両エージェントを対戦状態にする |
| メッセージフロー | 1. サーバーが `|request|` で始まるメッセージを送信 2. 各 `EnvPlayer`(P0/P1) がメッセージを解析して `battle` を更新し、PokemonEnv へフラグ付きで送信 3. PokemonEnv は受け取った行動を `EnvPlayer` 経由でサーバへ送信 4. サーバが結果を返す |

* 各 `request` には昇順の `rqid` が付与され、乱序で届くことがある
* 同一ターンに複数の `request` が送られることがある

---

## 3. 同期 / 非同期処理

* 外部: `poke_env` は **asyncio**‐ベースで WebSocket を管理  
* PokemonEnv API: **同期的** (`reset()`, `step()`)
 * Multi‑Agent 対応に伴い、`step()` は dict 形式を返す*
  * 例:
    ```python
    obs = {"player_0": np.ndarray, "player_1": np.ndarray}
    action = {"player_0": int, "player_1": int}
    reward = {"player_0": float, "player_1": float}
    ```

* 手順
  1. 非同期処理は poke-env が保持する `POKE_LOOP` イベントループを利用し、同期 API からは `asyncio.run_coroutine_threadsafe(coro, POKE_LOOP)` でタスクを登録し `future.result()` で待機する
  2. バトル開始待ちや状態待ちは `asyncio.Queue.get()` や `asyncio.Event.wait()` を `asyncio.wait_for()` と組み合わせて実施し、ビジーウェイトを行わない
  3. `reset()` で `battle_against()` を呼び、対戦を開始
  4. `PokemonEnv`は`reset()`内で`EnvPlayer`からメッセージが来るのを待機
  5. `EnvPlayer`は`|teampreview|`のメッセージが届いたら`PokeonEnv`に`my_team: List`と`opp_team: List`を渡して、チーム選択を要求(この時点ではBattleオブジェクトは空である)
  6. `PokemonEnv`は`reset()`の戻り値として`state`と`info`を`Agent`にわたす(`info`で`Agent`にチームプレビュー要求。現時点でstateは未実装なのでダミーを渡す。)
  7. `Agent`は`info`の情報からチーム選択が呼び出されたことを理解して、`step(choose_team(state))`を実行
  8. `PokemonEnv`はチーム選択を受け取り`EnvPlayer`に送信
  9. `EnvPlayer`はチーム選択をサーバに送信して新しい`request`を待つ
  10. `request`が発生したら`EnvPlaer`は`battle`オブジェクトを更新し、`PokemonEnv`に`battle`オブジェクトとフラグやキューを`PokemonEnv` にわたして`action`を待機する(ここで初めてbattleオブジェクトが更新)
  11. `PokemonEnv`は`Agent`にStateObserverを使って作成した情報ベクトルと、`action_helper.py`の`get_available_actions_with_details`で作成した選択可能な行動マスクを送信する。
  12. `Agent`は`choose_move(state,mask)`で行動を選択して、`step(action)`を呼ぶ
  13. `PokemonEnv`は`action`をキューに投入して、次の`request`フラグを待つ
  14. `EnvPlayer`(`poke-env`)は`action`を`battleorder`に変換してShowdownサーバに送信して、次の`request`を待つ
  15. `EnvPlayer`は次の`request`が来たら`battle`を更新して`PokemonEnv`に渡して、再度`action`を待機する
  16. `PokemonEnv`は`step(action)`の戻り値として`Agent`に`state(observation)`と`reward`,`done`(エピソード終了判定),`info`(未実装)を返す
  17. `Agent`は`受け取った情報から行動を選択して次の`step(action)`を呼ぶ

* 注意
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
* 観測は各プレイヤー視点で計算したベクトルを dict にまとめて返す

```text
observation = {
  "player_0": concat(
    own_active_stats,
    own_bench_1, own_bench_2,
    opp_active_stats,
    ...,
    global_field_info
  )
  "player_1":concat(...) #2p目線で計算
}
```

---

## 5. 行動空間

```
action_spaces = {
    "player_0": Discrete(10), # index 0‑9
    "player_1": Discrete(10)
}
```
`step()` には `{"player_0": idx0, "player_1": idx1}` の形で行動を渡す。

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

    participant Agent0 as Agent-0
    participant Agent1 as Agent-1
    participant PokemonEnv
    participant EnvP0 as poke-env/P0
    participant EnvP1 as poke-env/P1
    participant Showdown

    Agent0->>PokemonEnv: reset()
    Agent1->>PokemonEnv: reset()
    PokemonEnv->>EnvP0: play_against()
    PokemonEnv->>EnvP1: play_against()
    Showdown-->>EnvP0: |request|
    Showdown-->>EnvP1: |request|
    EnvP0->>PokemonEnv: battle update
    EnvP1->>PokemonEnv: battle update
    PokemonEnv->>Agent0: observe, mask
    PokemonEnv->>Agent1: observe, mask
    Agent0->>PokemonEnv: step(action0)
    Agent1->>PokemonEnv: step(action1)
    PokemonEnv->>EnvP0: battleorder0
    PokemonEnv->>EnvP1: battleorder1
    EnvP0->>Showdown: /choose
    EnvP1->>Showdown: /choose
    Showdown-->>EnvP0: |request|
    Showdown-->>EnvP1: |request|
    ...
```

---

## 7. 報酬・エピソード終了

| 状況 | terminated["player_0"] | terminated["player_1"] | reward["player_0"] | reward["player_1"] |
| --- | --- | --- |
| 自分が勝利 | True | +1 |
| 相手が勝利 | True | -1 | +1 |
| ターン > MAX_TURNS | True (truncated) | 0 | 0 |
| 途中ターン | False | 0 |

報酬は `{"player_0": float, "player_1": float}` 形式で返る。

---

## 8. 実装ノート

* **遅延インポート**: `poke_env` は `reset()` 内でインポート
* **EnvPlayer**: 行動アルゴリズムは外部エージェントに委任
* **チームプレビュー**: `Agent.choose_team()` でチーム選択を行い `/choose team` を送信（デフォルトはランダム3匹選出）
* **再利用接続**: 各エピソード開始時に `reset_battles()`
* **step 待機処理**: `asyncio.wait_for(queue.get(), timeout)` を用いて待ち合わせ、ビジーウェイトを避ける
* **close() 実装**: `POKE_LOOP` 上のタスクをキャンセルし、キューの `join()` 後にリソースを解放する
* **依存**: `poke-env>=0.9`, Showdown server (localhost:8000)
* **Multi‑Agent dict API**: 観測・行動・報酬・terminated/truncated・info はすべて `"player_0"`, `"player_1"` キー付き dict
* **対戦組み合わせ**: 同一 MapleAgent の重み共有 or スナップショット固定など、訓練シナリオに応じて差し替え可能

---

## 9. 参考コードスニペット

```python
# 環境ベクトル取得
state: np.ndarray = state_observer.observe(battle)

# 行動マスク取得
mask, mapping = action_helper.get_available_actions(battle)

# 行動インデックス -> BattleOrder
order = action_helper.action_index_to_order(env_player, battle, idx)
next_state, reward, terminated, truncated, info = env.step({
    "player_0": idx0,
    "player_1": idx1,
})
```

---

### 変更履歴
- 2025-06-12 Multi-Agent API 追加

### End of File
