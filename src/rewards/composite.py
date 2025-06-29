from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

try:  # PyYAML may not be installed in minimal environments
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback when PyYAML is missing
    yaml = None

from . import RewardBase, HPDeltaReward


class CompositeReward(RewardBase):
    """複数のサブ報酬を合成するクラス。"""

    DEFAULT_REWARDS: Mapping[str, Callable[[], RewardBase]] = {
        "hp_delta": HPDeltaReward,
    }

    def __init__(self, config_path: str, reward_map: Mapping[str, Callable[[], RewardBase]] | None = None) -> None:
        self.config_path = config_path
        self.reward_map = dict(self.DEFAULT_REWARDS)
        if reward_map is not None:
            self.reward_map.update(reward_map)
        self.rewards: Dict[str, RewardBase] = {}
        self.weights: Dict[str, float] = {}
        self.last_values: Dict[str, float] = {}
        self._load_config()

    def _load_config(self) -> None:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            text = ""

        if yaml is not None:
            cfg = yaml.safe_load(text) or {}
        else:
            cfg = self._parse_simple_yaml(text)
        reward_cfg = cfg.get("rewards", {})
        for name, params in reward_cfg.items():
            if not isinstance(params, Mapping):
                continue
            if not params.get("enabled", True):
                continue
            factory = self.reward_map.get(name)
            if factory is None:
                continue
            self.rewards[name] = factory()
            self.weights[name] = float(params.get("weight", 1.0))
            self.last_values[name] = 0.0

    def _parse_simple_yaml(self, text: str) -> Dict[str, Any]:
        """Very small YAML parser used when PyYAML is unavailable."""
        result: Dict[str, Any] = {}
        current: Dict[str, Any] | None = None
        subsection: Dict[str, Any] | None = None
        for line in text.splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            key, _, rest = line.strip().partition(":")
            rest = rest.strip()
            if indent == 0:
                current = result.setdefault(key, {})
                subsection = None
            elif indent == 2 and current is not None:
                if rest:
                    current[key] = self._convert_value(rest)
                    subsection = None
                else:
                    subsection = {}
                    current[key] = subsection
            elif indent == 4 and subsection is not None:
                subsection[key] = self._convert_value(rest)
        return result

    @staticmethod
    def _convert_value(value: str) -> Any:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            num = float(value)
            return int(num) if num.is_integer() else num
        except ValueError:
            return value

    def reset(self, battle: Any | None = None) -> None:
        for r in self.rewards.values():
            r.reset(battle)
        for key in self.last_values:
            self.last_values[key] = 0.0

    def calc(self, battle: Any) -> float:
        total = 0.0
        for name, r in self.rewards.items():
            value = float(r.calc(battle))
            self.last_values[name] = value
            total += value * self.weights.get(name, 1.0)
        return float(total)


__all__ = ["CompositeReward"]
