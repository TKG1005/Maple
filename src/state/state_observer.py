# src/state/state_observer.py

import numpy as np
import yaml

# import time # timeは現在使われていないようなのでコメントアウトまたは削除してよい
from poke_env.environment.abstract_battle import AbstractBattle

# PokemonType のインポートが不足している可能性があるため追加 (エンコーダーなどで利用する場合)
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.move_category import MoveCategory  # MoveCategoryも追加


class StateObserver:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            self.spec = yaml.safe_load(f)
        # _build_encoders は get_observation_dimension でも利用する可能性があるため、先に初期化
        self.encoders = self._build_encoders(self.spec)
        self.opp_total_estimate = 3  # 敵の手持ちの初期値

    def observe(self, battle: AbstractBattle) -> np.ndarray:
        state = []
        # battle が None の場合や、必要な属性がない場合に StateObserver がエラーにならないように、
        # _build_context や _extract が安全にデフォルト値を返す必要があります。
        # Gymnasiumの reset() から初期状態を得る際、Battleオブジェクトがまだ完全に準備できていない可能性も考慮。
        if battle is None:
            # Battle オブジェクトが None の場合、デフォルトの観測値を返すかエラーを出すか。
            # ここでは、デフォルト値で埋めた観測ベクトルを返すことを試みる。
            # ただし、get_observation_dimension() が正確な次元を返すため、
            # この observe(None) が呼ばれるケースは限定的かもしれない。
            # もし呼ばれるなら、デフォルト値で観測ベクトルを構築するロジックが必要。
            # print("Warning: observe called with None battle object. Returning default observation.")
            # return np.zeros(self.get_observation_dimension(), dtype=np.float32) #次元数分の0配列を返すなど
            # あるいは、StateObserverの設計として observe(None) を許容しない場合はエラーを送出してもよい。
            raise ValueError(
                "StateObserver.observe() called with None battle object, which is not supported for actual observation generation."
            )

        context = self._build_context(battle)

        for group, features in self.spec.items():
            for key, meta in features.items():
                raw_default = meta.get("default", 0)
                try:
                    default_val = (
                        eval(raw_default)
                        if isinstance(raw_default, str)
                        else raw_default
                    )
                except Exception:
                    default_val = raw_default

                val = self._extract(meta["battle_path"], context, default_val)

                # self.encoders の初期化は __init__ で行われているはず
                enc_func = self.encoders.get(
                    (group, key),
                    lambda x: (
                        [float(x)] if not isinstance(x, list) else [float(i) for i in x]
                    ),
                )  # デフォルトエンコーダもリストを返すように
                encoded_val = enc_func(val)

                # デバッグ用printは条件を絞るか、詳細ログレベルで管理した方が良い
                # if battle.turn == 1: # 例としてターン1の時だけ出力
                #    print(f"Debug observe: {key} raw='{val}' encoded='{encoded_val}'")

                # if battle.turn == 1:
                #     print(f"Debug observe: {key} raw='{val}' encoded='{encoded_val}'")

                state.extend(
                    encoded_val if isinstance(encoded_val, list) else [encoded_val]
                )

        return np.array(state, dtype=np.float32)

    def get_observation_dimension(self) -> int:
        """
        state_spec.yml に基づいて観測ベクトルの総次元数を計算します。
        エンコーダーの出力次元を考慮します。
        """
        dimension = 0
        # エンコーダーが実際に返すリストの長さを確認するために、
        # 各特徴量に対してダミーデータでエンコーダーを一度実行してみるのが最も確実です。
        # このメソッドは初期化時に一度だけ呼ばれる想定なので、多少処理が重くても許容範囲。

        # _build_encodersで作成されたエンコーダ関数を実際に使って次元を計算
        for group, features in self.spec.items():
            for key, meta in features.items():
                # エンコーダを取得
                # self.encodersのキーは (group, key)
                encoder_func = self.encoders.get((group, key))
                if encoder_func is None:
                    # 'identity' や未定義の場合、デフォルトでは1次元と仮定
                    # ただし、デフォルトエンコーダがリストを返す場合もあるので注意
                    # ここでは、デフォルトエンコーダが常に単一のfloatをリストに入れたものを返すと仮定して1とする
                    print(
                        f"Warning: No specific encoder found for {group}.{key}. Assuming 1 dimension."
                    )
                    dimension += 1
                    continue

                # エンコーダにダミーデータ（デフォルト値など）を渡して出力の長さを確認
                # metaからデフォルト値を取得
                raw_default = meta.get("default", 0)
                try:
                    # YAMLで '[1,0]' のようにリスト形式で書かれたデフォルト値も評価
                    default_value_for_test = (
                        eval(raw_default)
                        if isinstance(raw_default, str)
                        else raw_default
                    )
                except Exception:
                    default_value_for_test = raw_default  # 数値やNoneなど

                # onehotエンコーダの場合、デフォルト値がエンコード後のリスト形式になっていることがある
                # それ以外の場合、エンコーダが処理できる型（None, int, float, str, Enumなど）のダミー値を渡す
                # 例： 'identity'なら数値、'onehot'ならカテゴリ文字列やNone

                encoder_type = meta.get("encoder", "identity")
                test_input = None  # デフォルトのテスト入力

                if encoder_type == "onehot":
                    # onehot の場合、デフォルト値がエンコード済みリストであるか、
                    # classes の最初の要素、あるいは "none" をテスト入力とする
                    if isinstance(
                        default_value_for_test, list
                    ):  # デフォルトが既にリストならその長さを採用できるが、エンコーダを通す方が確実
                        test_input = (
                            default_value_for_test  # これをエンコーダがどう扱うか
                        )
                    elif meta.get("classes"):
                        test_input = meta["classes"][0]  # 最初のクラスでテスト
                    else:  # classesがないonehotは通常ありえないが、フォールバック
                        test_input = "none"
                elif encoder_type == "linear_scale" or encoder_type == "identity":
                    # 数値を期待するエンコーダには0やデフォルト値（数値化可能なもの）
                    try:
                        test_input = float(
                            default_value_for_test
                            if not isinstance(default_value_for_test, list)
                            else 0
                        )
                    except (ValueError, TypeError):
                        test_input = 0.0  # フォールバック
                else:  # 不明なエンコーダタイプ
                    test_input = default_value_for_test  # そのまま渡してみる

                try:
                    encoded_output = encoder_func(test_input)
                    if isinstance(encoded_output, (list, np.ndarray)):
                        dimension += len(encoded_output)
                    else:  # スカラー値が返ってきた場合 (エンコーダの実装による)
                        dimension += 1
                except Exception as e:
                    print(
                        f"Error while testing encoder for {group}.{key} with input '{test_input}': {e}"
                    )
                    # エラー時はフォールバックとして1次元加算（あるいは設定に基づいてエラーを出す）
                    if meta.get("encoder") == "onehot" and meta.get("classes"):
                        dimension += len(meta.get("classes"))  # onehotならクラス数
                    else:
                        dimension += 1
                    print(
                        f"Warning: Assuming 1 dimension for {group}.{key} due to encoder test error."
                    )

        if dimension == 0:
            raise ValueError(
                "Calculated observation dimension is 0. Check state_spec.yml and StateObserver.get_observation_dimension()."
            )
        return dimension

    def _build_context(self, battle: AbstractBattle) -> dict:
        # (既存の _build_context メソッドは変更なし)
        ctx = {"battle": battle}
        # active_pokemon や opponent_active_pokemon が None の場合を考慮
        my_team = list(battle.team.values()) if battle.team else []
        active = next(
            (p for p in my_team if p.active), None
        )  # None の場合のデフォルトを追加
        ctx["active"] = active
        ctx["active_sorted_moves"] = (
            sorted(active.moves.values(), key=lambda m: m.id)
            if active and active.moves
            else []
        )

        bench = (
            [p for p in my_team if not p.active] if active else my_team
        )  # activeがNoneなら全員bench扱い(要件次第)
        ctx["bench1"] = bench[0] if len(bench) > 0 else None
        ctx["bench2"] = bench[1] if len(bench) > 1 else None

        opp_team = list(battle.opponent_team.values()) if battle.opponent_team else []
        self.opp_total_estimate = max(self.opp_total_estimate, len(opp_team))
        opp_alive_seen = sum(1 for p in opp_team if not p.fainted)
        unknown_remaining = max(0, self.opp_total_estimate - len(opp_team))

        opp_active = next(
            (p for p in opp_team if p.active), None
        )  # None の場合のデフォルトを追加
        ctx["opp_active"] = opp_active
        opp_bench = [p for p in opp_team if not p.active] if opp_active else opp_team
        ctx["opp_bench1"] = opp_bench[0] if len(opp_bench) > 0 else None
        ctx["opp_bench2"] = opp_bench[1] if len(opp_bench) > 1 else None
        ctx["my_alive_count"] = sum(1 for p in my_team if not p.fainted)
        ctx["opp_alive_count"] = opp_alive_seen + unknown_remaining

        return ctx

    def _extract(self, path: str, ctx: dict, default):
        # (既存の _extract メソッドは変更なし)
        try:
            # pathがNoneや空文字の場合も考慮
            if not path:
                return default
            return eval(
                path,
                {
                    "PokemonType": PokemonType,
                    "MoveCategory": MoveCategory,
                    "AbstractBattle": AbstractBattle,
                },
                ctx,
            )  # Enum型をevalのスコープに追加
        except (AttributeError, TypeError, IndexError, NameError, SyntaxError):
            # print(f"Debug extract: Path '{path}' failed with error '{e}'. Returning default '{default}'.")
            return default
        except Exception:  # その他の予期せぬエラー
            # print(f"Debug extract: Path '{path}' failed with unexpected error '{e}'. Returning default '{default}'.")
            return default

    def _build_encoders(self, spec: dict):
        # (既存の _build_encoders メソッドは概ねそのままで良いが、出力が常にリストになるように調整を検討)
        # onehotエンコーダのデフォルト値の扱いを修正
        enc = {}
        for group, features in spec.items():
            for key, meta in features.items():
                enc_key = (group, key)
                kind = meta.get("encoder", "identity")
                # default値は eval せずに文字列のまま _onehot_encoder_simple に渡す
                raw_default_for_lambda = meta.get("default", "0")  # evalしない

                if kind == "identity":
                    # 出力をリスト[float]に統一
                    enc[enc_key] = lambda x, d=raw_default_for_lambda: [
                        float(
                            x if x is not None else eval(d) if isinstance(d, str) else d
                        )
                    ]
                elif kind == "onehot":
                    classes = meta.get("classes", [])
                    # _onehot_encoder_simple を呼び出すlambda
                    enc[enc_key] = (
                        lambda val, cls_list=classes, default_str=raw_default_for_lambda: self._onehot_encoder_simple(
                            val, cls_list, default_str
                        )
                    )
                elif kind == "linear_scale":
                    lo, hi = meta.get("range", [0, 1])
                    # evalできる形式のデフォルト値文字列を期待
                    enc[enc_key] = (
                        lambda x, lower=lo, upper=hi, d_str=raw_default_for_lambda: [
                            (
                                ((float(x) - lower) / (upper - lower))
                                if x is not None and upper > lower
                                else (
                                    float(eval(d_str))
                                    if isinstance(d_str, str)
                                    else float(d_str)
                                )
                            )
                        ]
                    )
                else:  # 未知のエンコーダタイプもリスト[float]で返す
                    enc[enc_key] = lambda x, d=raw_default_for_lambda: [
                        float(
                            x if x is not None else eval(d) if isinstance(d, str) else d
                        )
                    ]
        return enc

    # _onehot_encoder_simple を StateObserver のメソッドとして定義
    def _onehot_encoder_simple(self, val, cls_list: list, default_str_val: str) -> list:
        """onehotエンコーディングを行い、結果をリストで返す"""
        value_to_encode_str = ""
        if val is None:
            value_to_encode_str = "none"
        elif isinstance(val, (PokemonType, MoveCategory)):  # Enum型の場合
            value_to_encode_str = val.name.lower()
        elif hasattr(val, "name") and isinstance(
            val.name, str
        ):  # 他のEnumっぽいオブジェクト
            value_to_encode_str = val.name.lower()
        else:
            value_to_encode_str = str(val).lower()

        if not cls_list:  # クラスリストが空の場合
            # default_str_valがエンコード済みリスト形式（例: '[0,0,1]'）ならそれを評価して返す
            try:
                default_as_list = eval(default_str_val)
                if isinstance(default_as_list, list):
                    return [float(i) for i in default_as_list]
            except Exception:
                pass
            return []  # 空のクラスリストに対するエンコードは空リスト

        encoded_list = [
            1.0 if value_to_encode_str == str(c).lower() else 0.0 for c in cls_list
        ]

        if sum(encoded_list) == 0:  # クラスリストにない値が来た場合
            try:
                default_as_list = eval(default_str_val)
                if isinstance(default_as_list, list) and len(default_as_list) == len(
                    cls_list
                ):
                    return [float(i) for i in default_as_list]
                # default_str_val が単一の値で、それがクラスリストにあればonehot化する試みも可能だが、
                # ここでは default_str_val がエンコード済みリストであることを期待する。
            except Exception:
                # evalに失敗した場合やリストでない場合、'none'を探す
                if "none" in [str(c).lower() for c in cls_list]:
                    return [1.0 if "none" == str(c).lower() else 0.0 for c in cls_list]
                # 'none'もなければ、すべて0のリストを返す
        return encoded_list
