# IPC制御メッセージ一覧（Controller → Wrapper）

本メモは、IPCBattleController から IPCClientWrapper に配送される「IPC制御メッセージ」の代表例を整理したものです。Showdown生テキスト（protocol）は除外し、Wrapper が種別判別で「IPC制御」と判断する対象のみを列挙します。実装では NDJSON（1行=1JSON）で送受信し、必要最小限のメタデータ（battle_id, target_player など）を併記します。

前提:
- Showdownプロトコル: type="protocol"（data に生テキスト）→ Wrapper は PSClient の listen/_handle_message に委譲
- IPC制御: 下記の通り（Wrapper 内で _is_ipc_control により判別して自己処理 or Controller へフォワード）

## 1) ライフサイクル通知

- battle_invitation
  - 目的: player_1 側に招待到着を通知
  - 例: {"type":"battle_invitation","battle_id":"...","role":"player_1","format":"gen9randombattle"}

- battle_created
  - 目的: Node 側でバトル確立済みの確定通知
  - 例: {"type":"battle_created","battle_id":"...","room_id":"battle-gen9randombattle-123"}

- battle_start
  - 目的: 対戦開始の合図（以降は Showdown プロトコルが流れる）
  - 例: {"type":"battle_start","battle_id":"..."}

- battle_end
  - 目的: 対戦終了の合図（勝敗・理由など）
  - 例: {"type":"battle_end","battle_id":"...","result":"p1","reason":"forfeit|timeout|normal"}

- shutdown
  - 目的: プロセス停止の事前通知（グレースフル）
  - 例: {"type":"shutdown","battle_id":"...","deadline_ms":2000,"reason":"maintenance"}

- restart
  - 目的: プロセス再起動の事前通知（必要に応じて backoff）
  - 例: {"type":"restart","battle_id":"...","backoff_ms":500}

## 2) 接続・疎通（ヘルスチェック）

- ping
  - 目的: 疎通確認／ラウンドトリップ測定
  - 例: {"type":"ping","battle_id":"...","seq":101,"ts":1733712345678}

- pong
  - 目的: ping 応答
  - 例: {"type":"pong","battle_id":"...","seq":101,"ts":1733712345690}

- ready
  - 目的: Controller 側の準備完了通知（Wrapper 側が listen 開始可能）
  - 例: {"type":"ready","battle_id":"..."}

## 3) 登録・識別

- assign_player
  - 目的: Wrapper に自身の論理ID（player_0/player_1）と Showdown ID（p1/p2）を明示
  - 例: {"type":"assign_player","battle_id":"...","player_id":"player_0","target_player":"p1"}

- player_registered / player_unregistered
  - 目的: 対向プレイヤーの接続・切断イベントのブリッジ
  - 例: {"type":"player_registered","battle_id":"...","player_id":"player_1"}

## 4) エラー・例外・タイムアウト

- error
  - 目的: 重大エラー通知（Node クラッシュ、JSON デコード失敗、ID 不一致など）
  - 例: {"type":"error","battle_id":"...","code":"NODE_CRASH","message":"exit code 1"}

- warning
  - 目的: 非致命的な警告（遅延、軽微な不整合）
  - 例: {"type":"warning","battle_id":"...","code":"SLOW_IO","message":"stdout drain delayed"}

- timeout
  - 目的: 特定操作のタイムアウト通知（create_battle 応答待ち等）
  - 例: {"type":"timeout","battle_id":"...","operation":"create_battle","deadline_ms":3000}

## 5) フロー制御・情報

- flow_control
  - 目的: 一時的な読み出し抑制/再開（背圧対策）
  - 例: {"type":"flow_control","battle_id":"...","action":"pause|resume"}

- info
  - 目的: 任意の情報通知（統計、ログ要約、デバッグヒント）
  - 例: {"type":"info","battle_id":"...","message":"node stdout reader started"}

- ack
  - 目的: 任意コマンドの受領確認（id で突合）
  - 例: {"type":"ack","battle_id":"...","cmd_id":"c-12345","ok":true}
