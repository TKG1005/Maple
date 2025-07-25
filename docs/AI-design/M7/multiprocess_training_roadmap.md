# Mapleトレーニング高速化ロードマップ：マルチプロセス化実装

## 概要

PythonのGIL（Global Interpreter Lock）制約により、現在のThreadPoolExecutorベースの並列処理では、CPUコアを十分に活用できていない問題を解決する。**ProcessPoolExecutorによる段階的なマルチプロセス化**により、poke-envの仕様を遵守しながら真の並列処理を実現し、学習スピードを大幅に向上させる。

## アプローチの統合方針

### Claude案 vs GPT案の比較
- **Claude案**: 大規模アーキテクチャ変更、Shared Memory、完全な分離
- **GPT案**: 最小限の変更、既存コードベースの活用、段階的実装

### 統合方針
**GPT案をベースとし、Claude案の構造化された設計要素を段階的に導入**する方針を採用。poke-envの`POKE_LOOP`制約を最優先に考慮し、既存のMultiServerManagerとの互換性を保持する。

## 現状分析

### 現在のGILボトルネック

#### ThreadPoolExecutor使用箇所
- `train.py:924`: 対戦エピソード実行（混合/単一対戦モード）
- `train.py:949`: セルフプレイエピソード実行
- 各スレッドでCPU集約的処理：モデル推論、状態処理、リワード計算

#### CPU集約的処理の特定
- `RLAgent.select_action()`: ニューラルネットワーク推論
- `StateObserver.observe()`: 状態特徴量計算（2160次元）
- `CompositeReward.calc()`: 報酬計算
- Damage Calculation: 288特徴量の実時間計算

#### async/await制約
- poke-envのWebSocket通信は非同期
- `asyncio.run_coroutine_threadsafe()`でスレッド間ブリッジ
- `POKE_LOOP`への依存

### パフォーマンス問題
- 環境数を増やしても、思ったほどCPU使用率が上がらない
- 学習スピードの伸びが頭打ちになりやすい（特にCPUが多いIntel環境で顕著）
- モデル推論、状態処理、通信すべての場面でGILが律速になる

## アーキテクチャ設計

### プロセス分離設計

```
Main Process (学習・更新)
├── モデル更新 (PyTorch)
├── 経験集約
└── TensorBoard/CSV出力

Worker Processes (環境実行)
├── Process 0: env[0-4] + models[copy]
├── Process 1: env[5-9] + models[copy]  
└── Process N: env[N*5-N*5+4] + models[copy]
```

### Communication Architecture

```python
# プロセス間通信
Main ↔ Workers:
  - Shared Memory: モデル状態辞書（PyTorch state_dict）
  - Queue: 経験データ（observations, actions, rewards, etc.）
  - Event: 同期制御（モデル更新完了、プロセス終了）
```

## 統合実装ロードマップ

### Phase 1: ProcessPoolExecutor基盤実装 (GPT案ベース)

#### Step 1-1: 現行コード並列処理部分の把握
**対象**: `train.py:924, 949` のThreadPoolExecutor使用箇所

**確認事項**:
- `envs`リスト内の各環境でrun_episode実行
- `init_env()`による環境生成処理
- `env.close()`処理とRLAgent生成箇所
- 既存のMultiServerManager連携

#### Step 1-2: ProcessPoolExecutorへの置き換え
**実装**: `train.py`の最小限変更

```python
# Before: ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=len(envs)) as executor:

# After: ProcessPoolExecutor  
with ProcessPoolExecutor(max_workers=len(envs)) as executor:
```

**重要制約**:
- プロセス数は従来と同じく並列環境数(`parallel`値)に設定
- Windows対応: `if __name__ == "__main__":`ガード追加

#### Step 1-3: ワーカー関数の設計とシリアライズ対応
**新規ファイル**: `src/train/process_worker.py`

**機能**:
- pickle可能な引数のみを受け取る
- 子プロセス内で環境・エージェント初期化
- poke-env独立イベントループ利用

**実装詳細**:
```python
def run_episode_process(
    env_config: dict,
    model_state_dict: dict,
    network_config: dict, 
    opponent_config: dict,
    server_config: dict,
    gamma: float,
    lam: float,
    device_name: str,
    record_init: bool = False
) -> tuple:
    """プロセス内でエピソード実行"""
    
    # 1. 子プロセス内で環境初期化（poke_envの独立POKE_LOOP利用）
    env = init_env(
        reward=env_config["reward"],
        team_mode=env_config["team_mode"],
        server_config=server_config,
        # ... other config
    )
    
    # 2. モデルパラメータから エージェント構築
    policy_net = create_policy_network(network_config)
    value_net = create_value_network(network_config)
    policy_net.load_state_dict(model_state_dict["policy"])
    value_net.load_state_dict(model_state_dict["value"])
    
    # 3. 対戦相手エージェント生成
    opponent_agent = create_opponent_agent(opponent_config)
    
    # 4. エピソード実行（既存run_episodeロジック）
    # ... episode execution ...
    
    # 5. 環境クリーンアップ
    env.close()
    
    # 6. pickle可能な結果を返却
    return (batch_data, total_reward, init_tuple, sub_totals, opponent_type)
```

### Phase 2: poke-env制約の適切な処理

#### Step 2-1: 独立イベントループの活用
**重要発見**: poke-envは各プロセスで独立した`POKE_LOOP`を持つ
- 子プロセス内で`from poke_env.concurrency import POKE_LOOP`
- 自動的にプロセス専用イベントループが起動
- WebSocket接続もプロセスごとに独立

#### Step 2-2: MultiServerManager統合
**活用**: 既存のサーバー分散システム

```python
# server_assignmentsを子プロセスに渡す
server_config = server_assignments[env_id]
executor.submit(
    run_episode_process,
    env_config=env_config,
    server_config=server_config,  # プロセス専用サーバー情報
    # ... other args
)
```

### Phase 3: 結果集約と学習更新の最適化

#### Step 3-1: 親プロセスでの結果集約
**実装**: 既存の集約ロジックを維持

```python
# ProcessPoolExecutorの結果取得
results = [f.result() for f in futures]

# 既存の集約処理をそのまま利用
batches = [res[0] for res in results]
reward_list = [res[1] for res in results]
# ... existing aggregation logic ...
```

#### Step 3-2: モデル更新とパラメータ配布
**改善点**: state_dictの効率的な配布

```python
# 現在のモデルパラメータを取得
current_model_state = {
    "policy": policy_net.state_dict(),
    "value": value_net.state_dict()
}

# 各プロセスに配布（次エピソード用）
for i in range(len(envs)):
    futures.append(executor.submit(
        run_episode_process,
        model_state_dict=current_model_state,  # 最新パラメータ
        # ... other args
    ))
```

### Phase 4: Claude案要素の段階的導入

#### Step 4-1: Shared Memory最適化 (Optional)
**将来拡張**: 大規模モデル用のShared Memory

```python
# Advanced: Shared Memory for large models
from src.utils.shared_model import SharedModelState

shared_model = SharedModelState(policy_net)
# プロセス間でモデル共有
```

#### Step 4-2: 高度な並列化 (Optional)
**将来拡張**: バッチ処理の並列化

- Experience replay buffer の並列処理
- Gradient computation の分散処理

### Phase 5: 検証と最適化

#### Step 5-1: マルチプロセス実行の検証
**性能指標**:
- CPU使用率: 25-40% → 80-95%
- エピソード/秒: 2-3x向上期待
- メモリ使用量の監視

#### Step 5-2: poke-env統合テスト
**検証項目**:
- 各プロセスでのWebSocket接続安定性
- POKE_LOOPリソースリークの確認
- MultiServerManagerとの連携検証

#### Step 5-3: クロスプラットフォーム対応
**対応項目**:
- Windows spawn方式の対応
- デバッグ時の例外処理改善
- ログ出力の集約機能

## 技術的課題と解決策

### 1. poke-env Asyncio制約
**問題**: `POKE_LOOP`のスレッド共有
**解決**: 各プロセスで独立event loop

### 2. PyTorchモデル共有
**問題**: CUDA tensorsの共有制約  
**解決**: CPU state_dictのみ共有、各プロセスでGPU転送

### 3. Server接続管理
**問題**: 同一ポートへの重複接続
**解決**: MultiServerManagerの拡張、プロセス専用ポート割り当て

### 4. メモリ効率
**問題**: プロセス毎のメモリ増加
**解決**: Shared Memory最大活用、軽量Workerプロセス

## パフォーマンス期待値

### 現状 vs マルチプロセス

**現状（ThreadPool）**:
- CPU使用率: 25-40% (GIL制約)
- 並列効率: 低（1コアで律速）

**マルチプロセス後**:
- CPU使用率: 80-95% (全コア活用)
- 並列効率: 3-4x改善期待
- エピソード/秒: 2-3x向上

## 実装優先度（統合版）

### Phase 1: 即座実装 (High Priority)
1. **ProcessPoolExecutor置き換え**: `train.py`の最小限変更
2. **Process Worker実装**: `src/train/process_worker.py`
3. **シリアライズ対応**: pickle可能な引数設計

### Phase 2: 安定化 (High Priority)  
4. **poke-env統合**: 独立POKE_LOOP活用
5. **MultiServerManager連携**: 既存インフラとの統合
6. **エラーハンドリング**: デバッグ・例外処理

### Phase 3: 最適化 (Medium Priority)
7. **パフォーマンステスト**: CPU使用率・速度向上検証
8. **メモリ効率化**: リソースリーク対策
9. **クロスプラットフォーム**: Windows対応完了

### Phase 4: 高度化 (Low Priority)
10. **Shared Memory**: 大規模モデル対応
11. **分散学習**: Gradient並列化
12. **Advanced Optimization**: さらなる高速化

## 重要な実装制約

### poke-env固有制約
1. **POKE_LOOP**: 各プロセスで独立インスタンス生成
2. **WebSocket**: プロセスごとに独立接続必要
3. **Battle Object**: pickle不可のため、child processで再生成必須

### 既存システム制約
1. **MultiServerManager**: 既存のserver_assignments活用
2. **StateObserver**: 2160次元状態空間の効率処理
3. **Team Caching**: 37.2x高速化との共存

### パフォーマンス制約
1. **pickle Overhead**: state_dict転送コスト
2. **Process Creation**: プールによる再利用
3. **Memory Usage**: プロセス間でのメモリ増加

## 実装ガイドライン

### 開発順序
1. 最小限のWorkerプロセス実装
2. Shared Memoryベースのモデル共有
3. 経験データ集約システム
4. poke-env分離対応
5. 既存train.pyとの統合
6. パフォーマンステスト・調整

### テスト戦略
- 単体テスト: 各コンポーネントの独立動作
- 統合テスト: プロセス間通信
- パフォーマンステスト: CPU使用率、学習速度
- 回帰テスト: 既存機能との互換性

### モニタリング
- CPU使用率の監視
- メモリ使用量の追跡
- エピソード実行時間の測定
- プロセス間通信のオーバーヘッド測定

## 期待される成果

このマルチプロセス化により、以下の改善が期待される：

1. **学習速度の大幅向上**: 2-3倍の高速化
2. **CPU資源の効率活用**: 使用率80-95%達成
3. **スケーラビリティの向上**: より多くの並列環境での効率的学習
4. **既存機能の保持**: 全ての現行機能を維持

これらの改善により、より短時間でより効果的なポケモンAIの学習が可能になる。