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

## 6. テストの実行

```bash
pytest -q
```

## 7. 関連ドキュメント

- [M4 backlog](AI-design/M4/M4_backlog.md)
- [PokemonEnv 技術仕様書](AI-design/PokemonEnv_Specification.md)
