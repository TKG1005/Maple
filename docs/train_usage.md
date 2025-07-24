# train.py の使い方

`train.py` は、自己対戦形式で強化学習を行うためのスクリプトです。基本的な起動は次の通りです。

```bash
python train.py [オプション]
```

主なオプション一覧を以下に示します。省略した場合は `config/train_config.yml` に定義された値が使用されます。

| オプション | 説明 |
|------------|------|
| `--config FILE` | 設定ファイルのパスを指定します（既定: `config/train_config.yml`） |
| `--episodes N` | 実行するエピソード数を指定します |
| `--save FILE` | 学習後のモデルを保存するファイルパス。保存されるファイルには方策ネットワークと価値ネットワーク両方の重みが含まれます |
| `--load-model FILE` | 既存のモデルファイルから学習を再開します。ファイル名から自動的にエピソード番号を抽出します |
| `--tensorboard` | TensorBoard ログを有効にします |
| `--algo {reinforce,ppo}` | 使用するアルゴリズムを選択します |
| `--ppo-epochs N` | 1 エピソード終了後の PPO 更新回数 |
| `--clip R` | PPO のクリップ率を指定します |
| `--gae-lambda L` | GAE における λ パラメータ |
| `--parallel N` | 並列実行する環境数 |
| `--reward STR` | 使用する報酬タイプを指定します（既定: `composite`） |
| `--reward-config FILE` | CompositeReward 用の YAML ファイルを指定（既定: `config/reward.yaml`） |
| `--team {default,random}` | チーム選択モード。`random` でランダムチームを使用 |
| `--teams-dir DIR` | ランダムチーム用のディレクトリを指定（既定: `config/teams`） |
| `--opponent STR` | 対戦相手のタイプを指定（`random`, `max`, `rule`, `self`） |
| `--opponent-mix STR` | 複数の対戦相手を比率で混合（例: `"random:0.3,max:0.3,self:0.4"`） |

例:

```bash
# 基本的な学習
python train.py \
    --episodes 100 \
    --algo ppo \
    --ppo-epochs 4 \
    --clip 0.2 \
    --gae-lambda 0.95 \
    --parallel 2 \
    --save model.pt

# チェックポイントから学習を再開
python train.py \
    --load-model checkpoints/checkpoint_ep5000.pt \
    --episodes 100 \
    --save model_continued.pt

# ランダムチームで学習
python train.py \
    --team random \
    --teams-dir config/teams \
    --episodes 50 \
    --save model_random_teams.pt

# 複数の対戦相手を混合して学習
python train.py \
    --opponent-mix "random:0.3,max:0.3,self:0.4" \
    --episodes 100 \
    --save model_mixed_opponents.pt
```

オプションを指定しない場合でも、`train_config.yml` の内容が自動的に読み込まれるため、
最小限の入力で学習を開始できます。
デフォルトでは `CompositeReward` が使用されており、
`config/reward.yaml` の各項目を編集することで簡単に報酬の重みを調整できます。

## Knockout 報酬の使用例

ポケモンを倒す・倒されるイベントに応じた報酬を与える `KnockoutReward` を単独で
利用する場合は、`--reward knockout` を指定します。

```bash
python train.py --reward knockout
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

## evaluate_rl.py の使い方

`evaluate_rl.py` は、学習済みモデルの性能評価を行うためのスクリプトです。

主なオプション一覧:

| オプション | 説明 |
|------------|------|
| `--model FILE` | 評価するモデルファイルのパス |
| `--models FILE1 FILE2` | 2つのモデルを直接対戦させる場合のモデルファイル |
| `--opponent STR` | 対戦相手のタイプ（`random`, `max`, `rule`） |
| `--n N` | 実行するバトル数 |
| `--team {default,random}` | チーム選択モード |
| `--teams-dir DIR` | ランダムチーム用のディレクトリ |
| `--replay-dir DIR` | リプレイ保存ディレクトリ |

例:

```bash
# 基本的な評価
python evaluate_rl.py --model checkpoints/checkpoint_ep14000.pt --opponent random --n 10

# ランダムチームで評価
python evaluate_rl.py --model checkpoints/checkpoint_ep14000.pt --opponent rule --team random --n 5

# 2つのモデルを直接対戦
python evaluate_rl.py --models checkpoints/model_a.pt checkpoints/model_b.pt --n 10

# リプレイ保存付きで評価
python evaluate_rl.py --model checkpoints/checkpoint_ep14000.pt --opponent max --team random --replay-dir my_replays --n 3
```
