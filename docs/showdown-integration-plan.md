# Pokemon Showdown 直接統合実装計画書

## 概要

本文書は、Pokemon ShowdownサーバーをMapleフレームワークに直接組み込み、WebSocket通信のボトルネックを解消し、将来的にバトルシミュレーション機能を実装するための計画書である。オンライン対戦とローカル高速訓練の両立を実現するデュアルモードシステムを提案する。

## 0. 現状システム理解（実装前必読）

### 0.1 現在のアーキテクチャ

#### WebSocket通信フロー
```
PokemonEnv (Gymnasium環境)
    ↓ 
EnvPlayer (poke-env Player拡張クラス) 
    ↓ WebSocket接続 (ws://localhost:8000/showdown/websocket)
Pokemon Showdown Server (Node.js)
    ↓ 
Battle Stream (sim/battle-stream.ts)
    ↓
Battle Engine (sim/battle.ts)
```

#### 非同期-同期ブリッジ
- `EnvPlayer.choose_move()`: 同期インターフェース
- 内部でWebSocket通信: 非同期処理
- `asyncio.run_coroutine_threadsafe()`: 変換レイヤー（ボトルネック）

### 0.2 重要なファイル構造

```
src/env/
├── pokemon_env.py          # メインのGymnasium環境
├── env_player.py           # WebSocket通信ブリッジ
└── custom_battle.py        # カスタムバトル拡張

src/utils/
├── server_manager.py       # マルチサーバー管理
└── team_cache.py          # チームキャッシュ (37.2x高速化)

pokemon-showdown/
├── sim/                   # バトルエンジン (TypeScript)
├── server/                # WebSocketサーバー
└── data/                  # ゲームデータ

config/
├── train_config.yml       # 訓練設定
└── reward.yaml           # 報酬設定
```

### 0.3 現在の通信プロトコル

#### Pokemon Showdown Protocol Messages
```
# 送信例 (Python → Showdown)
>p1 move 1|tackle
>p2 switch 2

# 受信例 (Showdown → Python)  
|move|p1a: Pikachu|Thunder Shock|p2a: Charmander
|switch|p2a: Wartortle|Wartortle, L50, M|100/100
|-damage|p2a: Charmander|85/100
```

#### JSON Wrapper (IPC用に必要)
```json
{
    "type": "battle_command",
    "battle_id": "battle-gen9randombattle-12345",
    "player": "p1", 
    "command": "move 1"
}
```

### 0.4 既存の設定管理

#### train_config.yml 構造
```yaml
# 現在のサーバー設定
pokemon_showdown:
  servers:
    - host: "localhost"
      port: 8000
    - host: "localhost" 
      port: 8001

# 並列環境設定
parallel: 25
```

#### 環境作成パターン
```python
# 現在の初期化
env = PokemonEnv(
    server_configuration=server_config,
    parallel=parallel,
    team=team_mode,
    log_level=log_level
)
```

### 0.5 パフォーマンス測定基準

#### 現在のプロファイリング結果
- `env_step`: 11.7% (WebSocket通信部分)
- `env_reset`: 2.2% 
- 平均エピソード時間: 15秒/1000ステップ

#### 測定コマンド
```bash
# 現在のベンチマーク
python train.py --profile --episodes 10 --parallel 5

# ローカルモード測定 (実装後)
python train.py --battle-mode local --profile --episodes 10
```

### 0.6 依存関係とバージョン

#### 重要な依存関係
- `poke-env`: Pokemon Showdown Python クライアント
- `asyncio`: 非同期処理ライブラリ
- `websockets`: WebSocket通信
- Node.js 16+: Pokemon Showdown要件

#### 互換性制約
- poke-env Player APIとの完全互換性が必要
- 既存のCustomBattleクラスのオーバーライドを維持
- チームロード/キャッシュシステムとの統合

### 0.7 エラーハンドリングパターン

#### 現在の実装
```python
# EnvPlayer.choose_move() でのタイムアウト処理
try:
    result = await asyncio.wait_for(
        self._choose_move_async(battle), 
        timeout=self.timeout
    )
except asyncio.TimeoutError:
    logger.error("Battle timeout for player %s", self.username)
    return self.random_move()
```

#### IPCで必要なエラーハンドリング
- Node.jsプロセスのクラッシュ検出
- パイプブロック/デッドロック回避
- JSON parse エラーの処理
- プロセス再起動メカニズム

## 1. 背景と目的

### 1.1 現状の課題
- **WebSocket通信オーバーヘッド**: 環境step処理の11.7%を占める
- **非同期-同期変換コスト**: `asyncio.run_coroutine_threadsafe`による追加レイテンシ
- **JSONシリアライゼーション**: プロトコルメッセージの変換コスト
- **スケーラビリティ**: 並列環境数増加に伴うサーバー管理の複雑性

### 1.2 目標
1. ローカル訓練時のWebSocket通信を排除し、直接的なプロセス間通信を実現
2. オンライン対戦のためのWebSocket通信モードを維持
3. バトル状態の保存・復元機能の実装
4. 任意の状態から1-3ターンのシミュレーション実行機能
5. ローカルモードで現在の11.7%のオーバーヘッドを2-3%に削減

### 1.3 追加要件
- JSON形式での情報伝達を維持（プロトコル互換性）
- オンラインモードとローカルモードの切り替え可能
- 既存のpoke-env互換性を保持

## 2. 技術的検討

### 2.1 実装アプローチの比較

| アプローチ | 実現可能性 | 開発工数 | メンテナンス性 | パフォーマンス | 互換性 |
|-----------|-----------|---------|---------------|---------------|--------|
| TypeScript→Python変換 | ⚠️ 中 | 3-6ヶ月 | 低 | 最高 | 要再実装 |
| Node.js子プロセス+IPC | ✅ 高 | 2-3週間 | 高 | 高 | 維持可能 |
| PyV8/PyMiniRacer | ⚠️ 中 | 2-3ヶ月 | 中 | 高 | 困難 |

### 2.2 推奨アプローチ
**Node.js子プロセス + IPC通信（デュアルモード）**を採用する。理由：
- オリジナルのTypeScriptコードをそのまま活用可能
- JSON形式を維持しつつ高速化を実現
- オンライン/ローカルモードの柔軟な切り替え
- Pokemon Showdownの更新に容易に追従可能
- 段階的な実装が可能

### 2.3 デュアルモードアーキテクチャ

#### 現在の構成（オンラインモード）
```
Maple (Python) → WebSocket → Pokemon Showdown Server (Node.js)
                    ↑                    ↓
                ネットワーク層        独立プロセス
```

#### 新しい構成（ローカルモード）
```
Maple (Python) → IPC (JSON) → Pokemon Showdown (Node.js子プロセス)
                    ↑                    ↓
              同一マシン内通信       Mapleが直接管理
```

## 3. 実装フェーズ

### Phase 1: デュアルモード通信システム (2-3週間)

#### 3.1.1 目標
- WebSocket通信とIPC通信の抽象化
- モード切り替え可能な通信インターフェース
- JSON形式でのプロトコル互換性維持

#### 3.1.2 実装内容

**通信インターフェースの抽象化**
```python
# src/sim/battle_communicator.py
from abc import ABC, abstractmethod
import json

class BattleCommunicator(ABC):
    """バトル通信の抽象インターフェース"""
    
    @abstractmethod
    async def connect(self) -> None:
        pass
    
    @abstractmethod
    async def send_message(self, message: dict) -> None:
        pass
    
    @abstractmethod
    async def receive_message(self) -> dict:
        pass

class WebSocketCommunicator(BattleCommunicator):
    """従来のWebSocket通信（オンライン対戦用）"""
    
    def __init__(self, url: str):
        self.url = url
        self.ws = None
    
    async def send_message(self, message: dict):
        await self.ws.send(json.dumps(message))

class IPCCommunicator(BattleCommunicator):
    """高速IPC通信（ローカル訓練用）"""
    
    def __init__(self):
        self.process = None
        
    async def connect(self):
        # Node.js子プロセスを起動
        self.process = await asyncio.create_subprocess_exec(
            'node', 'sim/ipc-battle-server.js',
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    
    async def send_message(self, message: dict):
        # JSON形式を維持（互換性のため）
        data = json.dumps(message) + '\n'
        self.process.stdin.write(data.encode())
        await self.process.stdin.drain()
```

**モード切り替え可能なEnvPlayer**
```python
# src/env/dual_mode_player.py
class DualModeEnvPlayer(EnvPlayer):
    """オンライン/ローカルモード対応プレイヤー"""
    
    def __init__(self, 
                 mode: str = "local",  # "local" or "online"
                 server_config: Optional[ServerConfiguration] = None,
                 **kwargs):
        self.mode = mode
        
        if mode == "local":
            # IPC通信用の設定
            self.communicator = IPCCommunicator()
            self._override_websocket_methods()
        else:
            # 従来のWebSocket通信
            super().__init__(server_configuration=server_config, **kwargs)
            self.communicator = WebSocketCommunicator(
                f"ws://{server_config.host}:{server_config.port}/showdown/websocket"
            )
```

**Node.js側のIPCサーバー**
```javascript
// sim/ipc-battle-server.js
const {BattleStream} = require('./battle-stream');
const readline = require('readline');

class IPCBattleServer {
    constructor() {
        this.battles = new Map();
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'create_battle':
                this.createBattle(message);
                break;
            case 'battle_command':
                // 既存のShowdownプロトコルと互換
                this.processBattleCommand(message);
                break;
        }
    }
    
    sendMessage(message) {
        // JSON形式で送信（プロトコル互換性維持）
        console.log(JSON.stringify(message));
    }
}
```

#### 3.1.3 成果物
- デュアルモード対応の`BattleCommunicator`インターフェース
- `WebSocketCommunicator`と`IPCCommunicator`実装
- `DualModeEnvPlayer`クラス
- Node.js側のIPCサーバー実装
- モード切り替えテストスイート

### Phase 2: 環境統合とモード管理 (1-2週間)

#### 3.2.1 目標
- PokemonEnvへのデュアルモード統合
- 設定ファイルとCLIでのモード切り替え
- パフォーマンスモニタリング

#### 3.2.2 実装内容

**PokemonEnvの拡張**
```python
# src/env/pokemon_env.py
class PokemonEnv:
    def __init__(self, 
                 ...,
                 battle_mode: str = "local",  # "local" or "online"
                 server_configuration: Optional[ServerConfiguration] = None):
        
        self.battle_mode = battle_mode
        
        # モードに応じて適切なプレイヤーを作成
        if battle_mode == "local":
            logger.info("Using local IPC mode for battles")
            self._init_local_mode()
        else:
            logger.info("Using online WebSocket mode for battles")
            self._init_online_mode(server_configuration)
```

**設定ファイル統合**
```yaml
# config/train_config.yml
battle_mode: "local"  # "local" or "online"

# ローカルモード設定
local_mode:
  max_processes: 10  # 最大子プロセス数
  process_timeout: 300  # プロセスタイムアウト（秒）
  reuse_processes: true  # プロセス再利用

# オンラインモード設定（従来通り）
pokemon_showdown:
  servers:
    - host: "localhost"
      port: 8000
```

**CLIインターフェース**
```python
# train.py
parser.add_argument(
    "--battle-mode",
    type=str,
    choices=["local", "online"],
    default="local",
    help="Battle communication mode (local IPC or online WebSocket)"
)
```

#### 3.2.3 成果物
- デュアルモード対応PokemonEnv
- 設定ファイルスキーマ更新
- CLIパラメータ追加
- モード別パフォーマンスベンチマーク

### Phase 3: バトル状態シリアライゼーション (2-3週間)

#### 3.3.1 目標
バトル状態の完全な保存・復元機能の実装（両モード対応）

#### 3.3.2 実装内容
[既存の内容を維持]

### Phase 4: シミュレーション機能 (3-4週間)

#### 3.4.1 目標
任意のバトル状態から複数ターンのシミュレーション実行（ローカルモード専用）

#### 3.4.2 実装内容
[既存の内容を維持]

## 4. 統合計画

### 4.1 既存システムとの統合
```python
# src/env/pokemon_env.py の拡張
class PokemonEnv:
    def __init__(self, ..., battle_mode: str = "local"):
        self.battle_mode = battle_mode
        
        if battle_mode == "local":
            self.simulator = EmbeddedSimulator()
            self.battle_sim = BattleSimulator(self.simulator)
        # オンラインモードでは従来のWebSocket実装を使用
```

### 4.2 移行戦略
1. **デフォルト設定**: 新規インストールはローカルモードをデフォルトに
2. **後方互換性**: 既存コードは`battle_mode="online"`で動作継続
3. **段階的移行**: 設定ファイルまたはCLIフラグで簡単に切り替え
4. **A/Bテスト**: 両モードでの性能比較とバリデーション

## 5. パフォーマンス目標

| メトリクス | オンラインモード | ローカルモード目標 | 改善率 |
|-----------|----------------|-----------------|--------|
| 通信遅延 | 10-15ms | 1-2ms | 90%削減 |
| スループット | 1,000 msg/s | 10,000 msg/s | 10倍 |
| 環境step時間に占める通信の割合 | 11.7% | 2-3% | 75%削減 |
| 1000ステップ実行時間 | 15秒 | 12秒 | 20%削減 |
| メモリ使用量 | 基準値 | +10% | 許容範囲 |

## 6. リスクと対策

### 6.1 技術的リスク
- **リスク**: Node.jsプロセスのメモリリーク
- **対策**: プロセスプール管理と定期的な再起動機構

### 6.2 互換性リスク
- **リスク**: poke-envライブラリとの非互換性
- **対策**: 最小限のオーバーライドと包括的なテストスイート

### 6.3 保守性リスク
- **リスク**: Pokemon Showdownの大幅な更新による非互換性
- **対策**: バージョン固定オプションと自動互換性チェック

## 7. スケジュール

| フェーズ | 期間 | 開始予定 | 完了予定 |
|---------|-----|---------|---------|
| Phase 1: デュアルモード | 3週間 | 2025年2月1日 | 2025年2月21日 |
| Phase 2: 環境統合 | 2週間 | 2025年2月22日 | 2025年3月7日 |
| Phase 3: シリアライゼーション | 3週間 | 2025年3月8日 | 2025年3月28日 |
| Phase 4: シミュレーション | 4週間 | 2025年3月29日 | 2025年4月25日 |
| 統合テスト | 2週間 | 2025年4月26日 | 2025年5月9日 |

## 8. 成功基準

1. ローカルモードでWebSocket通信オーバーヘッドを75%以上削減
2. オンラインモードとの完全な互換性維持
3. JSON形式でのプロトコル互換性100%
4. バトル状態の100%正確な保存・復元
5. 3ターンシミュレーションを50ms以内で実行（ローカルモード）
6. 既存のテストスイートの100%パス
7. 両モードでの並列実行時の安定性維持

## 9. 実装チェックリスト

### Phase 1: デュアルモード実装準備
- [ ] 現在のWebSocket通信フローの詳細理解
- [ ] EnvPlayer.choose_move()の内部実装確認  
- [ ] Pokemon Showdownプロトコルメッセージフォーマット確認
- [ ] 既存のasyncio使用パターン調査
- [ ] IPCプロトタイプの作成とテスト

### Phase 2: 技術検証項目
- [ ] Node.js子プロセス起動/終了テスト
- [ ] JSON serialization/deserializationのパフォーマンス測定
- [ ] プロセス間通信のレイテンシ測定
- [ ] エラーハンドリングのテストケース作成
- [ ] メモリ使用量の比較測定

### Phase 3: 統合テスト項目
- [ ] 既存のテストスイートでの後方互換性確認
- [ ] モード切り替えの動作確認
- [ ] 並列環境での安定性テスト
- [ ] 長時間実行時のメモリリーク検出
- [ ] パフォーマンスベンチマークの実行

## 10. 開発環境セットアップ

### 前提条件
```bash
# Node.js 16+ がインストール済みであること
node --version  # v16.0.0+

# Python依存関係の確認
pip install -r requirements.txt

# Pokemon Showdownサーバーのテスト起動
cd pokemon-showdown
node pokemon-showdown
```

### 開発用コマンド
```bash
# 現在のシステムでの基準測定
python train.py --episodes 5 --parallel 5 --profile

# IPC実装後の比較測定
python train.py --battle-mode local --episodes 5 --parallel 5 --profile

# テストスイートの実行
pytest tests/ -v
pytest test/ -m slow  # 統合テスト
```

## 11. トラブルシューティング

### よくある問題
1. **Node.jsプロセスが起動しない**
   - Node.jsバージョンの確認
   - pokemon-showdownディレクトリの確認
   - 必要な依存関係の確認

2. **IPC通信がブロックする**
   - バッファサイズの調整
   - デッドロック検出の実装
   - タイムアウト値の調整

3. **JSON parsing エラー**
   - メッセージの改行文字確認
   - エンコーディングの確認
   - 不完全なメッセージの処理

### デバッグ手法
```python
# IPC通信のデバッグ
import logging
logging.getLogger('ipc_communicator').setLevel(logging.DEBUG)

# プロセス監視
import psutil
process = psutil.Process(node_process.pid)
print(f"Memory: {process.memory_info()}")
```

## 12. 参考資料

### Pokemon Showdown関連
- [Pokemon Showdown Protocol](pokemon-showdown/PROTOCOL.md)  
- [Battle Stream Documentation](pokemon-showdown/sim/SIM-PROTOCOL.md)
- [Pokemon Showdown API](https://github.com/smogon/pokemon-showdown)

### poke-env関連
- [poke-env Documentation](https://poke-env.readthedocs.io/)
- [Player Class API](https://poke-env.readthedocs.io/en/stable/player.html)
- [Battle Class API](https://poke-env.readthedocs.io/en/stable/battle.html)

### 実装例
- `src/env/env_player.py`: 現在のWebSocket実装
- `src/utils/server_manager.py`: プロセス管理の実装例
- `train.py`: 設定管理とCLIパターン

---

*最終更新: 2025年1月29日*
*作成者: Maple開発チーム*