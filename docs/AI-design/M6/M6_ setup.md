# Maple M7 セットアップガイド

このドキュメントでは、PPO アルゴリズム導入後の Maple を利用して学習と評価を行う手順をまとめます。

## 1. 事前準備

- Python 3.11 以上がインストールされていること
- `git clone` したリポジトリ直下で次のコマンドを実行して依存ライブラリを導入します

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pokémon Showdown サーバを別端末で起動しておく必要があります。

```bash
npx pokemon-showdown
```

## 2. PPO での学習

`train.py` を `--algo ppo` オプション付きで実行すると、PPO アルゴリズムを用いた自己対戦学習を開始できます。
主なハイパーパラメータは以下の通りです。

| オプション | 説明 |
|------------|------|
| `--ppo-epochs N` | 1 エピソード終了後の更新回数 |
| `--clip R` | 方策確率比のクリップ率 |
| `--gae-lambda L` | GAE における λ の値 |
| `--parallel N` | 並列実行する環境数 |

例:

```bash
python train.py --episodes 100 --algo ppo --ppo-epochs 4 \
    --clip 0.2 --gae-lambda 0.95 --parallel 2 --save ppo_model.pt
```

## 3. アルゴリズムの切り替え

`--algo` 引数には `reinforce` と `ppo` が指定できます。省略時は `ppo` が利用されます。既存の REINFORCE 手法で学習したい場合は次のように指定します。

```bash
python train.py --algo reinforce --episodes 50 --save reinforce.pt
```

## 4. 学習モデルの評価

学習済みモデルは `evaluate_rl.py` で対戦させて評価できます。

```bash
python evaluate_rl.py --model ppo_model.pt --n 5
```

2 つのモデルを比較する場合は `--models` を使います。

```bash
python evaluate_rl.py --models model_a.pt model_b.pt --n 10
```

## 5. 学習結果の比較

`logs/` に保存された学習ログから `plot_compare.py` を実行すると、REINFORCE と PPO の報酬推移を 1 つのグラフにまとめた `compare.png` を生成します。

```bash
python plot_compare.py
```

生成された画像を確認することで、アルゴリズム間の性能差を把握できます。

---

以上で環境構築から PPO 学習・評価までの基本的な流れが完了します。詳細なオプションは `docs/train_usage.md` も参照してください。
