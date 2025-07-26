# Maple - Windows Setup Guide

Windows環境でのMaple（Pokemon強化学習フレームワーク）のセットアップとサーバー管理ガイド

## 🚀 Windows用サーバー管理スクリプト

Windows環境でPokemon Showdownサーバーを簡単に管理できる2つのスクリプトを提供しています：

### 1. バッチファイル版 (`showdown.bat`)

コマンドプロンプト用のスクリプト

```cmd
# サーバー起動（5台）
scripts\showdown.bat start 5

# サーバー起動（10台）
scripts\showdown.bat start 10

# 設定ファイルに基づく自動起動
scripts\showdown.bat quick

# サーバー状況確認
scripts\showdown.bat status

# 全サーバー停止
scripts\showdown.bat stop

# ヘルプ表示
scripts\showdown.bat help
```

### 2. PowerShell版 (`showdown.ps1`)

PowerShell用のスクリプト（より詳細な出力とエラーハンドリング）

```powershell
# サーバー起動（5台）
.\scripts\showdown.ps1 start 5

# サーバー起動（10台）
.\scripts\showdown.ps1 start 10

# 設定ファイルに基づく自動起動
.\scripts\showdown.ps1 quick

# サーバー状況確認
.\scripts\showdown.ps1 status

# 全サーバー停止
.\scripts\showdown.ps1 stop

# ヘルプ表示
.\scripts\showdown.ps1 help
```

## 📋 必要な環境

### 前提条件

1. **Node.js** (v16以上推奨)
   - [Node.js公式サイト](https://nodejs.org/)からダウンロード・インストール
   - インストール後、コマンドプロンプトで `node --version` を実行して確認

2. **Python** (3.8以上)
   - [Python公式サイト](https://www.python.org/)からダウンロード・インストール
   - PATHに追加することを確認

3. **Git** (オプション)
   - [Git for Windows](https://gitforwindows.org/)

### Pokemon Showdownの準備

プロジェクトルートに `pokemon-showdown` ディレクトリが存在することを確認してください。

## 🛠️ セットアップ手順

### 1. 依存関係のインストール

```cmd
# Python依存関係
pip install -r requirements.txt

# Pokemon Showdown依存関係（pokemon-showdownディレクトリ内で実行）
cd pokemon-showdown
npm install
cd ..
```

### 2. 設定ファイルの確認

`config/train_config.yml` でサーバー設定を確認：

```yaml
pokemon_showdown:
  servers:
    - host: "localhost"
      port: 8000
      max_connections: 25
    - host: "localhost"
      port: 8001
      max_connections: 25
    # ... 必要な数だけ追加
```

### 3. サーバー起動

#### コマンドプロンプト使用
```cmd
# 5台のサーバーを起動
scripts\showdown.bat start 5

# 状況確認
scripts\showdown.bat status
```

#### PowerShell使用
```powershell
# PowerShell実行ポリシーの設定（初回のみ）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 5台のサーバーを起動
.\scripts\showdown.ps1 start 5

# 状況確認
.\scripts\showdown.ps1 status
```

### 4. 訓練実行

```cmd
# 基本的な訓練実行
python train.py --parallel 5 --episodes 10 --device cpu

# マルチプロセス使用（推奨）
python train.py --parallel 10 --episodes 50 --use-multiprocess --device cpu
```

## 📊 サーバー管理

### サーバー状況の確認

```cmd
# バッチファイル版
scripts\showdown.bat status

# PowerShell版
.\scripts\showdown.ps1 status
```

出力例：
```
🔍 Pokemon Showdown Server Status
📂 Project root: C:\path\to\Maple
🔌 Checking port range: 8000-8010
═══════════════════════════════════════════════════════════════════════════════

💻 System Information:
   Node.js: v18.17.0
   Current time: 2025/07/27 10:30:00

🔍 Checking servers on ports 8000-8010...
─────────────────────────────────────────────────────────────────
🟢 Port 8000: RUNNING (Managed)
   PID: 12345
   Log: logs\showdown_logs\showdown_server_8000.log
   Log size: 2048 bytes | Last activity: 2025/07/27 10:25:00

🟢 Port 8001: RUNNING (Managed)
   PID: 12346
   Log: logs\showdown_logs\showdown_server_8001.log
   Log size: 1536 bytes | Last activity: 2025/07/27 10:25:00
```

### ログファイル監視

```cmd
# バッチファイル - ログ表示
type logs\showdown_logs\showdown_server_8000.log

# PowerShell - リアルタイムログ監視
Get-Content "logs\showdown_logs\showdown_server_8000.log" -Wait
```

### サーバー停止

```cmd
# 全サーバー停止
scripts\showdown.bat stop

# または PowerShell版
.\scripts\showdown.ps1 stop
```

## 🔧 トラブルシューティング

### よくある問題と対処法

#### 1. Node.jsが見つからない
```
❌ Node.js not found. Please install Node.js first.
```
**対処法**: Node.jsをインストールし、PATHに追加してください。

#### 2. ポートが既に使用されている
```
❌ Server #1 failed to start on port 8000
```
**対処法**: 
- 他のプロセスがポートを使用していないか確認
- `netstat -an | findstr :8000` でポート使用状況を確認
- 必要に応じて異なるポートを使用

#### 3. PowerShell実行ポリシーエラー
```
実行ポリシーによってこのスクリプトの実行が拒否されました
```
**対処法**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 4. 権限エラー
**対処法**:
- 管理者権限でコマンドプロンプト/PowerShellを実行
- Windowsファイアウォールの設定を確認

### ログファイルの場所

- **サーバーログ**: `logs\showdown_logs\showdown_server_[PORT].log`
- **PIDファイル**: `logs\pids\showdown_[PORT].pid`

### パフォーマンス最適化

1. **並列数の調整**:
   - `parallel環境数 ≤ サーバー数` を維持
   - 例: 10並列環境なら10台のサーバーを起動

2. **マルチプロセス使用**:
   ```cmd
   python train.py --parallel 10 --use-multiprocess
   ```

3. **リソース監視**:
   - タスクマネージャーでCPU/メモリ使用量を監視
   - 必要に応じて並列数を調整

## 📁 ディレクトリ構造

```
Maple/
├── scripts/
│   ├── showdown.bat      # Windows バッチファイル版
│   ├── showdown.ps1      # Windows PowerShell版
│   └── showdown          # Unix/Linux版（既存）
├── logs/
│   ├── showdown_logs/    # サーバーログファイル
│   └── pids/             # PIDファイル
├── config/
│   └── train_config.yml  # 設定ファイル
└── pokemon-showdown/     # Pokemon Showdownサーバー
```

## 🚀 基本的なワークフロー

1. **初期セットアップ**:
   ```cmd
   # 依存関係インストール
   pip install -r requirements.txt
   cd pokemon-showdown && npm install && cd ..
   ```

2. **サーバー起動**:
   ```cmd
   scripts\showdown.bat start 5
   ```

3. **訓練実行**:
   ```cmd
   python train.py --parallel 5 --episodes 10
   ```

4. **サーバー停止**:
   ```cmd
   scripts\showdown.bat stop
   ```

## 💡 ヒント

- PowerShell版の方が詳細なログとエラー処理を提供
- 大規模訓練（parallel > 10）では `--use-multiprocess` を使用
- サーバー数は並列環境数以上に設定
- 定期的にログファイルをクリーンアップ

## 📞 サポート

問題が発生した場合：

1. `scripts\showdown.bat status` または `.\scripts\showdown.ps1 status` で状況確認
2. ログファイル（`logs\showdown_logs\`）を確認
3. Node.js、Python、依存関係のバージョンを確認
4. GitHubのIssueで報告

---

Windows環境でのMaple使用を快適にお楽しみください！ 🎮✨