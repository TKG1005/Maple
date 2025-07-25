# Pokemon Showdown Server Management Scripts

複数のPokemon Showdownサーバーを簡単に起動・管理するためのスクリプト集です。

## 🚀 クイックスタート

```bash
# 5つのサーバーを起動（ポート8000-8004）
./scripts/showdown start

# 設定ファイルに基づいて自動起動
./scripts/showdown quick

# サーバーの状態確認
./scripts/showdown status

# 全サーバーを停止
./scripts/showdown stop
```

## 📋 利用可能なコマンド

### メインユーティリティ (`./scripts/showdown`)

最も簡単な使用方法：

```bash
./scripts/showdown <command> [arguments]
```

**コマンド一覧:**
- `start [num] [port]` - サーバー起動 (デフォルト: 5サーバー、ポート8000～)
- `stop [start] [end]` - サーバー停止 (デフォルト: ポート8000-8010をチェック)
- `status [start] [end]` - サーバー状態確認
- `restart [num] [port]` - サーバー再起動
- `quick` - train_config.ymlの設定に基づいて自動起動
- `logs [port]` - ログ表示 (特定ポートまたは全て)
- `help` - ヘルプ表示

### 個別スクリプト

#### `start_showdown_servers.sh`
複数のPokemon Showdownサーバーを起動します。

```bash
./scripts/start_showdown_servers.sh [サーバー数] [開始ポート]

# 例
./scripts/start_showdown_servers.sh 5 8000    # 5つのサーバーをポート8000-8004で起動
./scripts/start_showdown_servers.sh 3 8010    # 3つのサーバーをポート8010-8012で起動
```

**特徴:**
- バックグラウンドで実行
- PIDファイルによるプロセス管理
- ログファイルの自動作成
- ポート使用状況の自動チェック
- train_config.ymlの設定例を自動表示

#### `stop_showdown_servers.sh`
実行中のPokemon Showdownサーバーを停止します。

```bash
./scripts/stop_showdown_servers.sh [開始ポート] [終了ポート]

# 例
./scripts/stop_showdown_servers.sh             # 全ての管理されたサーバーを停止
./scripts/stop_showdown_servers.sh 8000 8004  # ポート8000-8004のサーバーを停止
```

#### `status_showdown_servers.sh`
サーバーの実行状況を詳細に表示します。

```bash
./scripts/status_showdown_servers.sh [開始ポート] [終了ポート]

# 例
./scripts/status_showdown_servers.sh           # ポート8000-8010をチェック
./scripts/status_showdown_servers.sh 8000 8005 # ポート8000-8005をチェック
```

**表示内容:**
- サーバーの実行状態 (RUNNING/STOPPED)
- プロセスID (PID)
- CPU・メモリ使用率
- ログファイル情報
- 管理状態 (スクリプトで管理されているか)

#### `restart_showdown_servers.sh`
サーバーを安全に再起動します。

```bash
./scripts/restart_showdown_servers.sh [サーバー数] [開始ポート]

# 例
./scripts/restart_showdown_servers.sh 5 8000  # 5つのサーバーを再起動
```

## 📁 ファイル構造

```
scripts/
├── showdown                    # メインユーティリティ
├── start_showdown_servers.sh   # サーバー起動
├── stop_showdown_servers.sh    # サーバー停止
├── status_showdown_servers.sh  # 状態確認
├── restart_showdown_servers.sh # 再起動
└── README.md                   # このファイル

logs/
├── showdown_pids/              # PIDファイル格納
│   ├── showdown_server_8000.pid
│   ├── showdown_server_8001.pid
│   └── ...
└── showdown_logs/              # ログファイル格納
    ├── showdown_server_8000.log
    ├── showdown_server_8001.log
    └── ...
```

## 🔧 実際の使用例

### 1. 開発環境での基本的な使用

```bash
# 5つのサーバーを起動
./scripts/showdown start

# 状態確認
./scripts/showdown status

# 学習を実行
python train.py --episodes 10 --parallel 20 --device cpu

# サーバー停止
./scripts/showdown stop
```

### 2. 大規模並列学習

```bash
# 20個のサーバーを起動（ポート8000-8019）
./scripts/showdown start 20 8000

# 状態確認
./scripts/showdown status 8000 8019

# train_config.ymlを更新してから学習を実行
python train.py --episodes 1000 --parallel 400 --device cpu
```

### 3. 設定ファイルベースの起動

```bash
# train_config.ymlの設定に基づいて自動起動
./scripts/showdown quick

# この場合、設定ファイルのサーバー数・ポート番号が自動で使用されます
```

### 4. ログの確認

```bash
# 特定ポートのログを表示
./scripts/showdown logs 8001

# 全サーバーのログを表示
./scripts/showdown logs

# リアルタイムログ監視
tail -f logs/showdown_logs/showdown_server_8000.log
```

## ⚠️ 注意事項

1. **ポート競合**: 他のアプリケーションが同じポートを使用していないか確認してください
2. **リソース制限**: 多数のサーバーを起動する場合は、システムリソースに注意してください
3. **ファイアウォール**: 必要に応じてファイアウォール設定を確認してください
4. **Node.js要件**: Pokemon Showdownの実行にはNode.jsが必要です

## 🐛 トラブルシューティング

### サーバーが起動しない場合

```bash
# ポート使用状況を確認
lsof -i :8000

# Node.jsのバージョン確認
node --version

# pokemon-showdownディレクトリの存在確認
ls -la pokemon-showdown/
```

### サーバーが応答しない場合

```bash
# プロセス確認
./scripts/showdown status

# 強制停止
./scripts/showdown stop

# 再起動
./scripts/showdown restart
```

### ログを確認したい場合

```bash
# エラーログを確認
./scripts/showdown logs [ポート番号]

# システムログを確認
ps aux | grep pokemon-showdown
```

## 🚀 train_config.yml設定例

スクリプトで起動したサーバー構成をtrain_config.ymlに反映する例：

```yaml
pokemon_showdown:
  servers:
    - host: "localhost"
      port: 8000
      max_connections: 25
    - host: "localhost"
      port: 8001
      max_connections: 25
    - host: "localhost"
      port: 8002
      max_connections: 25
    - host: "localhost"
      port: 8003
      max_connections: 25
    - host: "localhost"
      port: 8004
      max_connections: 25
```

この設定により、複数サーバーへの負荷分散が自動的に機能します。