# train_selfplay.py の使い方

`train_selfplay.py` は、自己対戦形式で強化学習を行うためのスクリプトです。基本的な起動は次の通りです。

```bash
python train_selfplay.py [オプション]
```

主なオプション一覧を以下に示します。省略した場合は `config/train_config.yml` に定義された値が使用されます。

| オプション | 説明 |
|------------|------|
| `--config FILE` | 設定ファイルのパスを指定します（既定: `config/train_config.yml`） |
| `--episodes N` | 実行するエピソード数を指定します |
| `--save FILE` | 学習後のモデルを保存するファイルパス。保存されるファイルには方策ネットワークと価値ネットワーク両方の重みが含まれます |
| `--tensorboard` | TensorBoard ログを有効にします |
| `--algo {reinforce,ppo}` | 使用するアルゴリズムを選択します |
| `--ppo-epochs N` | 1 エピソード終了後の PPO 更新回数 |
| `--clip R` | PPO のクリップ率を指定します |
| `--gae-lambda L` | GAE における λ パラメータ |
| `--parallel N` | 並列実行する環境数 |
| `--reward-config FILE` | CompositeReward 用の YAML ファイルを指定 |

例:

```bash
python train_selfplay.py \
    --episodes 100 \
    --algo ppo \
    --ppo-epochs 4 \
    --clip 0.2 \
    --gae-lambda 0.95 \
    --parallel 2 \
    --save model.pt
```

オプションを指定しない場合でも、`train_config.yml` の内容が自動的に読み込まれるため、
最小限の入力で学習を開始できます。

## Knockout 報酬の使用例

ポケモンを倒す・倒されるイベントに応じた報酬を与える `KnockoutReward` を単独で
利用する場合は、`--reward knockout` を指定します。

```bash
python train_selfplay.py --reward knockout
```

複数の報酬を組み合わせたいときは、CompositeReward 用の YAML で各項目を有効にするだけで
利用できます。例として `config/reward.yaml` では次のように設定します。

```yaml
rewards:
  hp_delta:
    weight: 1.0
    enabled: true
  knockout:
    weight: 1.0
    enabled: true
  turn_penalty:
    weight: 1.0
    enabled: true
```
