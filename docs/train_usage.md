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

例:

```bash
python train_selfplay.py --episodes 100 --algo ppo --clip 0.2 --gae-lambda 0.95 --save model.pt
```

オプションを指定しない場合でも、`train_config.yml` の内容が自動的に読み込まれるため、
最小限の入力で学習を開始できます。
