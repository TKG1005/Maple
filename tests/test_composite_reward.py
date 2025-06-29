import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import CompositeReward, RewardBase


class ConstReward(RewardBase):
    def __init__(self, value: float) -> None:
        self.value = value
        self.reset_called = False

    def reset(self, battle: object | None = None) -> None:
        self.reset_called = True

    def calc(self, battle: object) -> float:
        return float(self.value)


def test_composite_reward(tmp_path: Path) -> None:
    yaml_path = tmp_path / "reward.yaml"
    yaml_text = (
        "rewards:\n"
        "  a:\n"
        "    weight: 2.0\n"
        "    enabled: true\n"
        "  b:\n"
        "    weight: 0.5\n"
        "    enabled: true\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    comp = CompositeReward(
        str(yaml_path),
        reward_map={"a": lambda: ConstReward(1.0), "b": lambda: ConstReward(1.0)},
    )
    comp.reset(None)
    assert all(r.reset_called for r in comp.rewards.values())
    reward = comp.calc(None)
    assert reward == 2.0 * 1.0 + 0.5 * 1.0
    assert comp.last_values["a"] == 1.0
    assert comp.last_values["b"] == 1.0
