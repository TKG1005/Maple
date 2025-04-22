
import numpy as np
import yaml
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.move import Move

class StateObserver:
    def __init__(self, yaml_path: str):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.spec = yaml.safe_load(f)
        self.encoders = self._build_encoders(self.spec)

    def observe(self, battle: AbstractBattle) -> np.ndarray:
        """Battleオブジェクトから状態ベクトルを生成"""
        state = []
        context = self._build_context(battle)
        for group, features in self.spec.items():
            for key, meta in features.items():
                val = self._extract(meta['battle_path'], context)
                enc = self.encoders.get((group, key), lambda x: x)
                encoded = enc(val)
                if isinstance(encoded, list):
                    state.extend(encoded)
                else:
                    state.append(encoded)
        return np.array(state, dtype=np.float32)

    def _build_context(self, battle: AbstractBattle) -> dict:
        """battle から必要なポケモンスロットを構築"""
        ctx = {'battle': battle}
        my_team = list(battle.team.values())
        active = next(p for p in my_team if p.active)
        bench = [p for p in my_team if not p.active]
        ctx['bench1'] = bench[0] if len(bench) > 0 else None
        ctx['bench2'] = bench[1] if len(bench) > 1 else None
        opp_team = list(battle.opponent_team.values())
        opp_bench = [p for p in opp_team if not p.active]
        ctx['opp_bench1'] = opp_bench[0] if len(opp_bench) > 0 else None
        ctx['opp_bench2'] = opp_bench[1] if len(opp_bench) > 1 else None
        ctx['my_alive_count'] = sum(1 for p in my_team if not p.fainted)
        ctx['opp_alive_count'] = sum(1 for p in opp_team if not p.fainted)
        return ctx

    def _extract(self, path: str, ctx: dict):
        """battle_path を評価して値を取り出す"""
        try:
            return eval(path, {}, ctx)
        except Exception:
            return 0

    def _build_encoders(self, spec: dict):
        """エンコーダ関数をフィールドごとに準備"""
        enc = {}
        for group, features in spec.items():
            for key, meta in features.items():
                enc_key = (group, key)
                kind = meta.get('encoder', 'identity')
                default = meta.get('default', 0)
                if kind == 'identity':
                    enc[enc_key] = lambda x, d=default: float(x) if x is not None else float(d)
                elif kind == 'onehot':
                    classes = meta.get('classes', [])
                    enc[enc_key] = lambda x, cls=classes, d=default: [1 if str(x).lower() == str(c).lower() else 0 for c in cls]
                elif kind == 'linear_scale':
                    lo, hi = meta.get('range', [0, 1])
                    enc[enc_key] = lambda x, l=lo, h=hi, d=default: (float(x) - l) / (h - l) if x is not None else float(d)
                else:
                    enc[enc_key] = lambda x, d=default: float(x) if x is not None else float(d)
        return enc
