
import numpy as np
import yaml
import time
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.move import Move

class StateObserver:
    def __init__(self, yaml_path: str):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            self.spec = yaml.safe_load(f)
        self.encoders = self._build_encoders(self.spec)
        self.opp_total_estimate = 3 #　敵の手持ちの初期値

    def observe(self, battle: AbstractBattle) -> np.ndarray:
        state = []
        context = self._build_context(battle)

        for group, features in self.spec.items():
            for key, meta in features.items():

                # ① default を YAML 文字列 → Python 値に直して渡す
                raw_default = meta.get("default", 0)
                try:
                    # YAML では '1' や '[1,0]' のように文字列で入っているので eval する
                    default_val = eval(raw_default) if isinstance(raw_default, str) else raw_default
                except Exception:
                    default_val = raw_default  # 数値や None のとき

                # ② _extract に default を渡す
                val = self._extract(meta["battle_path"], context, default_val)
                print(val)

                enc = self.encoders.get((group, key), lambda x: x)
                encoded = enc(val)
                state.extend(encoded if isinstance(encoded, list) else [encoded])
        
        return np.array(state, dtype=np.float32)
    
    def _build_context(self, battle: AbstractBattle) -> dict:
        ctx = {'battle': battle}
        my_team = list(battle.team.values())
        active = next(p for p in my_team if p.active)
        ctx['active'] = active
        ctx['active_sorted_moves'] = sorted(active.moves.values(), key=lambda m: m.id)
        bench = [p for p in my_team if not p.active]
        ctx['bench1'] = bench[0] if len(bench) > 0 else None
        ctx['bench2'] = bench[1] if len(bench) > 1 else None
        
        # ----------- 相手側 -----------------------------
        opp_team = list(battle.opponent_team.values())
        #  これまでに見えた敵の合計が初期推定 (3) を超えたら上書き
        self.opp_total_estimate = max(self.opp_total_estimate, len(opp_team))
        #  気絶していない敵の数を数える
        opp_alive_seen = sum(1 for p in opp_team if not p.fainted)
        #  未確認ポケモンの数 = 推定総数 − これまでに見えた数
        unknown_remaining = self.opp_total_estimate - len(opp_team)
        unknown_remaining = max(0, unknown_remaining)  # 念のため

        
        opp_active = next(p for p in opp_team if p.active)
        ctx['opp_active'] = opp_active
        opp_bench = [p for p in opp_team if not p.active]
        ctx['opp_bench1'] = opp_bench[0] if len(opp_bench) > 0 else None
        ctx['opp_bench2'] = opp_bench[1] if len(opp_bench) > 1 else None
        ctx['my_alive_count'] = sum(1 for p in my_team if not p.fainted)
        # 推定残数 = 生存確認 + 未確認
        ctx['opp_alive_count'] = opp_alive_seen + unknown_remaining
        
        return ctx

    def _extract(self, path: str, ctx: dict, default):
        try:
            return eval(path, {}, ctx)
        except Exception:
            return default
        
        
    def _build_encoders(self, spec: dict):
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
