# IPCBattle廃止計画

## 概要

IPCBattleクラスを廃止し、IPCClientWrapperにPSClient互換機能を統合することで、poke-envとの統合を簡素化し、アーキテクチャの重複を解消する。

## 背景・目的

### 現在の問題
- **機能重複**: IPCBattleとIPCClientWrapperで同様の機能（戦闘コマンド送信、状態取得）が重複
- **アーキテクチャ複雑性**: showdown ↔ IPC ↔ IPCBattle ↔ poke-env の複雑な統合
- **責任分離不明確**: どちらがメッセージ処理の主体か不明
- **保守性問題**: 変更時に複数クラスの更新が必要

### 目指す構造
```
showdown ↔ websocket ↔ ps_client (オンライン)
showdown ↔ IPC ↔ IPCClientWrapper (ローカル)
         ↓
    poke-env _handle_message() 統合
```

## 要件定義

### 1. IPCBattle廃止
- IPCBattleクラスとその関連機能を完全削除
- IPCCommunicatorとの通信はIPCClientWrapperが直接担当

### 2. IPCClientWrapper機能拡張
- **PSClient互換インターフェース**: AccountConfiguration、ServerConfiguration対応
- **接続確立・認証**: poke-envのPSClientと同等の機能
- **listen()メソッド**: PSClientのlisten()を模倣
- **メッセージ解析**: showdownプロトコルとIPC制御メッセージの判別
- **poke-env統合**: _handle_message()以降はオリジナルのpoke-env処理を使用

### 3. DualModePlayer統合
- AccountConfigurationオブジェクトを利用してIPCClientWrapperに対戦初期化をリクエスト
- WebSocketオーバーライド方法の変更

## 修正影響範囲

### 削除対象ファイル
```
src/sim/ipc_battle.py
src/sim/ipc_battle_factory.py  
tests/test_ipc_battle.py
```

### 修正対象ファイル

#### 主要修正
```python
# src/env/dual_mode_player.py - IPCClientWrapper拡張
class IPCClientWrapper:
    def __init__(self, account_configuration: AccountConfiguration, server_configuration, ...)
    async def listen(self)  # 新規実装
    async def log_in(self, split_message)  # 認証処理
    def _parse_message_type(self, message) -> tuple[bool, dict]  # メッセージ解析
    def _forward_to_poke_env(self, message)  # poke-env統合

# src/env/dual_mode_player.py - DualModeEnvPlayer修正  
class DualModeEnvPlayer:
    def _initialize_communicator(self)  # IPCClientWrapper初期化変更
    def _establish_full_ipc_connection(self)  # 認証フロー追加
    def _override_websocket_methods(self)  # ps_client置換方法変更
```

#### 軽微修正
```python
# src/env/pokemon_env.py
- IPCBattleFactory import削除
- _create_ipc_battles()メソッド削除 (lines 738-798)

# src/damage/calculator.py
- IPCBattle関連エラーメッセージ修正

# simple_teampreview_debug.py  
- IPCBattle関連分析コード削除
```

## 実装計画

### Phase 1: IPCClientWrapper拡張 🔧
**目標**: PSClient互換機能の実装

#### 1.1 AccountConfiguration互換性追加
```python
class IPCClientWrapper:
    def __init__(self, 
                 account_configuration: AccountConfiguration,
                 server_configuration: ServerConfiguration = None,
                 communicator: BattleCommunicator = None,
                 logger: logging.Logger = None):
        self.account_configuration = account_configuration
        self.server_configuration = server_configuration
        self.communicator = communicator
        self.logger = logger
        self.logged_in = asyncio.Event()  # poke-env互換
```

#### 1.2 listen()メソッド実装
```python
async def listen(self):
    """PSClient.listen()を模倣したIPC版実装"""
    try:
        # IPC接続確立
        if not await self.communicator.is_alive():
            await self.communicator.connect()
        
        # メッセージループ開始
        while True:
            message = await self.communicator.receive_message()
            await self._handle_message(message)
            
    except Exception as e:
        self.logger.error(f"IPC listen failed: {e}")
        raise
```

#### 1.3 認証機能追加
```python
async def log_in(self, split_message: List[str]):
    """PSClient.log_in()を模倣した認証処理"""
    # IPC環境ではchallstrは不要だが、互換性のため実装
    assertion = ""  # IPC環境では認証不要
    
    # ログイン成功として処理
    self.logged_in.set()
    self.logger.info(f"IPC login successful: {self.account_configuration.username}")

async def wait_for_login(self):
    """PSClient.wait_for_login()互換"""
    await self.logged_in.wait()
```

#### 1.4 メッセージ解析機能実装
```python
def _parse_message_type(self, message) -> tuple[bool, dict]:
    """showdownプロトコルとIPC制御メッセージを判別"""
    if not isinstance(message, dict):
        return False, {}
    
    msg_type = message.get("type")
    if msg_type == "protocol":
        # showdownプロトコルメッセージ
        return True, message
    elif msg_type in ["battle_created", "player_registered", "battle_end", "error"]:
        # IPC制御メッセージ  
        return False, message
    else:
        return False, message

async def _handle_message(self, message):
    """メッセージ処理のメインディスパッチャ"""
    is_showdown, parsed = self._parse_message_type(message)
    
    if is_showdown:
        # showdownプロトコル → poke-env _handle_message()
        protocol_data = parsed.get("data", "")
        await self._forward_to_poke_env(protocol_data)
    else:
        # IPC制御メッセージ処理
        await self._handle_ipc_control_message(parsed)
```

### Phase 2: DualModePlayer統合 🔗
**目標**: IPCClientWrapperとDualModePlayerの統合

#### 2.1 初期化方法変更
```python
def _initialize_communicator(self) -> None:
    if self.mode == "local":
        # 従来: IPCClientWrapper(communicator, logger)
        # 新方式: IPCClientWrapper(account_configuration, server_configuration, ...)
        self._communicator = CommunicatorFactory.create_communicator(...)
        
        # AccountConfigurationを取得（Player基底クラスから）
        account_config = getattr(self, 'account_configuration', None)
        server_config = getattr(self, 'server_configuration', None)
        
        self.ipc_client_wrapper = IPCClientWrapper(
            account_configuration=account_config,
            server_configuration=server_config,
            communicator=self._communicator,
            logger=self._logger
        )
```

#### 2.2 WebSocketオーバーライド変更
```python
def _override_websocket_methods(self) -> None:
    # 従来: self.ps_client = IPCClientWrapper(self._communicator, self._logger)
    # 新方式: self.ps_client = self.ipc_client_wrapper (Phase 2.1で作成済み)
    
    self._original_ps_client = getattr(self, 'ps_client', None)
    self.ps_client = self.ipc_client_wrapper
    
    # poke-env内部メソッドのオーバーライドは従来通り
    self._override_poke_env_internals()
```

#### 2.3 接続確立フロー修正
```python
def _establish_full_ipc_connection(self) -> None:
    async def establish_connection():
        # IPCClientWrapper.listen()開始
        listen_task = asyncio.create_task(self.ipc_client_wrapper.listen())
        
        # 認証完了待機
        await self.ipc_client_wrapper.wait_for_login()
        
        # ping-pongテスト（従来通り）
        await self._test_ipc_connection()
        
        return True
```

### Phase 3: IPCBattle削除 🗑️
**目標**: IPCBattle関連コードの完全削除

#### 3.1 ファイル削除
```bash
rm src/sim/ipc_battle.py
rm src/sim/ipc_battle_factory.py
rm tests/test_ipc_battle.py
```

#### 3.2 pokemon_env.py修正
```python
# 削除対象
from src.sim.ipc_battle_factory import IPCBattleFactory  # 削除

def _create_ipc_battles(self, ...):  # メソッド全体削除 (lines 738-798)

# _run_ipc_battle()メソッドからIPCBattleFactory呼び出し削除
```

#### 3.3 その他ファイル修正
```python
# src/damage/calculator.py
# エラーメッセージ修正
"Root cause: Pokemon or Move objects are not properly initialized with required stats/attributes."

# simple_teampreview_debug.py  
# IPCBattle関連の分析コード削除
```

### Phase 4: テスト・ドキュメント更新 📚
**目標**: 品質保証と文書化

#### 4.1 テストコード更新
```python
# tests/test_dual_mode_communication.py
class TestIPCClientWrapper:
    async def test_psclient_compatibility(self):
        """PSClient互換性テスト"""
        
    async def test_authentication_flow(self):
        """認証フローテスト"""
        
    async def test_message_parsing(self):
        """メッセージ解析テスト"""
```

#### 4.2 統合テスト
- オンラインモードとローカルモードの動作同等性確認
- AccountConfiguration連携確認
- showdownプロトコル処理確認

#### 4.3 ドキュメント更新
- CLAUDE.md更新: IPCBattle削除、IPCClientWrapper拡張の記載
- アーキテクチャ図更新
- 使用例更新

## 期待される効果

### アーキテクチャ簡素化
- **統合ポイント削減**: IPCBattle → IPCClientWrapper統合により1層削減
- **責任分離明確化**: IPCClientWrapperがIPC通信の単一責任点
- **poke-env統合簡素化**: _handle_message()以降は標準poke-env処理

### 保守性向上
- **機能重複解消**: 戦闘コマンド送信、状態取得の実装統一
- **変更影響範囲縮小**: IPCClientWrapper修正のみで完結
- **テスト簡素化**: 単一クラスのテストに集約

### 拡張性向上
- **PSClient互換**: 既存のpoke-envエコシステムと自然に統合
- **新機能追加容易**: IPCClientWrapper内で完結
- **デバッグ簡素化**: 単一のメッセージフロー

## リスク・注意点

### 実装リスク
- **poke-env内部依存**: _handle_message()の内部実装に依存
- **メッセージ解析複雑性**: showdown vs IPC判別の正確性
- **既存機能互換性**: 現在のIPC機能の完全な移植

### 対策
- **段階的実装**: Phase分けによる漸進的移行
- **テスト充実**: 各Phase完了時の動作確認
- **ロールバック準備**: 既存コードのバックアップ保持

## 作業再開時のチェックリスト

### 開始前確認
- [ ] 現在のIPCBattle、IPCClientWrapper動作確認
- [ ] poke-envバージョン互換性確認
- [ ] テスト環境準備

### Phase 1完了確認
- [ ] IPCClientWrapper拡張機能実装完了
- [ ] AccountConfiguration互換性確認
- [ ] listen()メソッド動作確認
- [ ] メッセージ解析機能テスト

### Phase 2完了確認  
- [ ] DualModePlayer統合完了
- [ ] WebSocketオーバーライド動作確認
- [ ] 認証フロー確認

### Phase 3完了確認
- [ ] IPCBattle関連ファイル削除完了
- [ ] インポートエラー解消確認
- [ ] 全体動作確認

### Phase 4完了確認
- [ ] テストコード更新・実行確認
- [ ] ドキュメント更新完了
- [ ] 最終統合テスト実行

---

**作成日**: 2025-01-05  
**更新日**: 2025-01-05  
**ステータス**: 計画段階  
**責任者**: システム設計チーム