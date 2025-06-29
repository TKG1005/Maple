# Step16 他アルゴリズム対応設計

## 目的
A2C や SAC など新しい強化学習アルゴリズムを導入する際に、既存コードを大幅に変更せず組み込めるよう拡張点を整理する。

## 現状の構成
- `src/agents/RLAgent` が `policy_net`、`value_net`、`optimizer`、`algorithm` を保持し、学習更新は `algorithm.update()` に委譲している。
- `src/algorithms/` には `BaseAlgorithm`、`ReinforceAlgorithm`、`PPOAlgorithm` が実装されている。
- オンポリシー前提のため、リプレイバッファの利用は想定されていない。

## 設計方針
1. **アルゴリズムインタフェースの一般化**
   - `BaseAlgorithm.update(model, optimizer, batch)` を拡張し、複数ネットワークやオプティマイザを扱えるようにする。
   - 例: `update(self, networks: Dict[str, nn.Module], optimizers: Dict[str, optim.Optimizer], batch: Dict[str, Tensor]) -> float`。
2. **ネットワーク管理の抽象化**
   - `RLAgent` が任意数のネットワークを登録できるよう `self.networks: Dict[str, nn.Module]` 形式で保持する。
   - 既存の `policy_net` / `value_net` は `networks["policy"]` / `networks["value"]` としてアクセスする。
3. **リプレイバッファの活用**
   - 現在 `ReplayBuffer` クラスが存在するが使用箇所が限定的。SAC などオフポリシー手法では必須となるため、
     `RLAgent` にバッファ操作 API を持たせ、学習スクリプト側でデータ収集とサンプリングを行う。
4. **離散・連続行動の両対応**
   - 行動空間に応じてネットワーク出力を切り替えられるよう、`PolicyNetwork` の派生クラスを用意する。
   - 例: 離散用 `DiscretePolicyNetwork`、連続用 `ContinuousPolicyNetwork`。
5. **オンポリシー / オフポリシーの切替え**
   - 学習ループ部分で ``on_policy`` フラグを設け、オンポリシーの場合は軌跡収集後即更新、オフポリシーの場合はバッファからサンプルして更新する。

## アルゴリズム別拡張ポイント
### A2C (Advantage Actor-Critic)
- **同期並列実行**: `train_selfplay.py` の `--parallel` オプションを利用し、複数環境から同時にデータを取得して平均勾配を計算する。
- **損失計算**: 方策ネットワークと価値ネットワークを同時に更新する。PPO実装の価値損失計算を再利用可能。
- **変更箇所**:
  - 新規 `A2CAlgorithm` を `src/algorithms` に追加。
  - `RLAgent.update()` が `networks` を渡すよう修正。

### SAC (Soft Actor-Critic)
- **リプレイバッファ**: 大規模バッファからランダムサンプルして学習。既存 `ReplayBuffer` を拡張し、`sample()` で複数ステップの遷移を返す。
- **複数Critic**: Qネットワークを2つ保持し、ターゲットネットワークも管理する必要がある。
- **連続行動対応**: `ContinuousPolicyNetwork` を利用し、ガウス分布から行動をサンプルする設計を想定。
- **変更箇所**:
  - `SACAlgorithm` を実装し、更新時に actor・critic・target の3種類のオプティマイザを用いる。
  - 学習スクリプトで ``on_policy=False`` とし、各ステップ終了時にバッファへ遷移を保存、一定ステップごとにバッチ学習を行う。

## 今後の進め方
1. 本ドキュメントの方針をもとに `BaseAlgorithm` と `RLAgent` のインタフェースを拡張する PR を作成する。
2. A2C, SAC それぞれ最小構成の実装を追加し、ユニットテストで更新処理が呼び出せることを確認する。
3. 既存スクリプトの CLI に `--algo a2c` などの選択肢を増やし、学習ログでアルゴリズム名を出力する。

以上により、新しいアルゴリズム追加時の変更範囲を限定し、将来の拡張を容易にする。
