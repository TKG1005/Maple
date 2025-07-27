# Windows Server Management Scripts

Windows環境でPokemon Showdownサーバーを管理するためのスクリプトです。

## 必要な環境

- Windows 10/11
- Node.js (v14以上)
- Python 3.7以上
- psutil (`pip install psutil`)
- PyYAML (`pip install pyyaml`)

## 使用方法

### バッチファイル (showdown.bat)

コマンドプロンプトから以下のコマンドを実行：

```cmd
# 5つのサーバーを起動（デフォルト）
scripts\showdown.bat start

# 指定した数のサーバーを起動
scripts\showdown.bat start 10

# 全サーバーを停止
scripts\showdown.bat stop

# サーバーの状態を確認
scripts\showdown.bat status

# train_config.ymlに基づいて自動起動
scripts\showdown.bat quick

# サーバーを再起動
scripts\showdown.bat restart 5
```

### Python ヘルパースクリプト (showdown_helper.py)

より詳細な制御が必要な場合：

```cmd
# 特定のポートでサーバーを起動
python scripts\showdown_helper.py start 8000

# 特定のポートのサーバーを停止
python scripts\showdown_helper.py stop 8000

# 詳細なステータスを表示
python scripts\showdown_helper.py status

# 設定から必要なサーバー数を取得
python scripts\showdown_helper.py config
```

## 機能

### showdown.bat
- **自動Python検出**: python, python3, py コマンドを自動検出
- **PIDファイル管理**: logs/pids/ にPIDファイルを保存
- **ログファイル**: logs/showdown_logs/ にサーバーログを保存
- **重複起動防止**: 既に起動しているサーバーはスキップ
- **一括管理**: 複数サーバーの起動/停止を一括実行
- **設定連携**: train_config.yml から必要なサーバー数を自動計算

### showdown_helper.py
- **クロスプラットフォーム対応**: Windows/Mac/Linuxで動作
- **プロセス管理**: psutilを使用した確実なプロセス制御
- **ポート監視**: 使用中のポートを検出
- **詳細なステータス**: プロセス情報、ログサイズ、最終更新時刻を表示

## ディレクトリ構造

```
Maple/
├── scripts/
│   ├── showdown.bat          # Windowsバッチファイル
│   ├── showdown_helper.py    # Pythonヘルパースクリプト
│   └── README_Windows.md     # このファイル
├── logs/
│   ├── pids/                 # PIDファイル保存場所
│   │   └── showdown_8000.pid
│   └── showdown_logs/        # サーバーログ保存場所
│       └── showdown_server_8000.log
└── config/
    └── train_config.yml      # 訓練設定ファイル
```

## トラブルシューティング

### Node.jsが見つからない
```
Error: Node.js not found. Please install Node.js
```
→ Node.jsをインストールしてPATHに追加してください

### Pythonが見つからない
```
Error: Python not found. Please install Python 3.x
```
→ Python 3.xをインストールしてPATHに追加してください

### ポートが使用中
```
Port 8000: IN USE (unmanaged)
```
→ 他のプロセスがポートを使用しています。`netstat -an | findstr :8000` で確認してください

### サーバーが起動しない
1. `pokemon-showdown` ディレクトリで `npm install` を実行
2. ログファイルを確認: `type logs\showdown_logs\showdown_server_8000.log`
3. ファイアウォールの設定を確認

## 注意事項

- 管理者権限は不要ですが、ファイアウォールの警告が表示される場合があります
- サーバーはバックグラウンドで実行されます
- 各サーバーは最大25の並列接続を処理できます
- ログファイルは自動的にローテーションされません（手動で削除が必要）