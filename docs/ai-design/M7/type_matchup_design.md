1. 新規クラス TypeMatchupFeatureExtractor の実装
目的
Battle オブジェクトからタイプ相性ダメージ倍率を計算し、1 次元ベクトルとして返す。

配置
src/state/type_matchup_extractor.py

主な処理

__init__(gen:int=9) で poke_env.data.GenData.from_gen(gen).type_chart を保持。

extract(battle: AbstractBattle) -> np.ndarray を実装。

自分のアクティブポケモンの技1〜4を state_observer と同様に sorted(active.moves.values(), key=lambda m: m.id)[:4] で取得。

相手ポケモンは battle.opponent_active_pokemon、opp_bench1、opp_bench2 を利用（存在しない場合は None とみなし 1.0 を返す）。

各技と各相手ポケモン(テラスタルしている場合はテラスタル後のタイプ）のタイプ組み合わせに対し PokemonType.damage_multiplier または pokemon.damage_multiplier(move) を使って倍率を求め、12 個の値に展開。

相手アクティブポケモンのタイプ一致技を仮定し、同ポケモンの type_1, type_2　の自分側（アクティブ + ベンチ1 + ベンチ2）へのダメージ倍率を計算して６個の値に展開。（type_2 が存在しない場合は 1.0)（テラスタル済みの場合はtype_1 を置き換える)

上記計算結果を 1D np.ndarray（計18要素）で返す。

2. StateObserver への統合
対象ファイル
src/state/state_observer.py

変更点

__init__ で TypeMatchupFeatureExtractor を生成し保持。

_build_context 内で self.feature_extractor.extract(battle) を呼び、結果を ctx["type_matchup_vec"] へ格納。

既存の spec 読み込み・エンコード処理がこのベクトル要素を参照できるようにする。

3. state_spec.yml の拡張
対象ファイル
config/state_spec.yml

追加内容例 (抜粋)

type_matchup:
  move1_vs_opp_active:
    dtype: float
    battle_path: type_matchup_vec[0]
    encoder: identity
    default: '1'
  move1_vs_opp_bench1:
    dtype: float
    battle_path: type_matchup_vec[1]
    encoder: identity
    default: '1'
  # ... 同様に move4_vs_opp_bench2 まで計12項目
  opp_stab1_vs_my_active:
    dtype: float
    battle_path: type_matchup_vec[12]
    encoder: identity
    default: '1'
  opp_stab1_vs_my_active:
    dtype: float
    battle_path: type_matchup_vec[13]
    encoder: identity
    default: '1'
  opp_stab1_vs_my_bench1:
    dtype: float
    battle_path: type_matchup_vec[14]
    encoder: identity
    default: '1'
  # ... 同様にopp_stab2_vs_my_bench2 まで計６項目
必要に応じて state_feature_catalog_temp - シート1.csv も更新する。

4. ユニットテスト追加
ファイル
tests/test_type_matchup_extractor.py

テスト内容

ダミー Battle とダミー Pokemon を用意し、特定のタイプ相性が分かる状況を構成。

TypeMatchupFeatureExtractor.extract() を実行してベクトル長が 15 であること、火→草など既知の倍率が正しく計算されることを検証。

ベンチポケモンが None の場合に 1.0 が返ることも確認。

5. ドキュメント更新
docs/AI-design/PokemonEnv_Specification.md などの状態空間説明箇所に新たな特徴量を追記し、計算方法・ベクトル位置を明記する。

6. 既存コードへの影響確認
PokemonEnv や各エージェントの状態取得処理は StateObserver.observe() に依存しているため、上記変更だけで自動的に新特徴量が取り込まれる想定。
万一バグが起きた場合は train_rl.py で簡単に学習を実行し、エラーの有無を確認する。

