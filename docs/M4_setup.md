# Maple M4 セットアップガイド

このドキュメントでは、Maple プロジェクトの M4 フェーズで必要となる環境構築手順をまとめます。開発を始める前に以下のステップを実施してください。

## 1. 前提条件

- Python 3.11 以上
- Node.js (Pokémon Showdown サーバ起動用)
- `git` がインストールされていること

## 2. リポジトリの取得

```bash
git clone https://github.com/yourname/Maple.git
cd Maple
```

## 3. 仮想環境の準備

```bash
python -m venv .venv
source .venv/bin/activate
```

## 4. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

## 5. Pokémon Showdown サーバの起動

別ターミナルで以下を実行してローカルサーバを立ち上げます。

```bash
npx pokemon-showdown
```

Showdown サーバの設定によっては、未登録ユーザーからのバトルメッセージを拒否
する場合があります。その際は環境変数 `PS_USERNAME0`/`PS_PASSWORD0` と
`PS_USERNAME1`/`PS_PASSWORD1` を設定して、登録済みアカウントでログインして
ください。

## 6. テストの実行

```bash
pytest -q
```

## PRNG と再現性について

ローカル実行時に `--seed` オプションで乱数シードを指定できますが、
Showdown サーバの PRNG は外部から制御できません。
そのため対戦の挙動を完全に再現することはできず、M5 のバックログに
ある「乱数シード制御」ステップはスキップされています。

## 7. 関連ドキュメント

- [M4 backlog](AI-design/M4/M4_backlog.md)
- [PokemonEnv 技術仕様書](AI-design/PokemonEnv_Specification.md)
