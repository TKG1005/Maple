"""Environment wrappers for compatibility."""

from __future__ import annotations

from typing import Any

from src.agents.MapleAgent import MapleAgent

import gymnasium as gym

try:  # Some tests stub out gymnasium
    GymWrapper = gym.Wrapper
except AttributeError:  # pragma: no cover - fallback for stubs
    GymWrapper = object

from .pokemon_env import PokemonEnv


class SingleAgentCompatibilityWrapper(GymWrapper):
    """Wrap :class:`PokemonEnv` for single-agent Gym interface."""

    def __init__(self, env: PokemonEnv) -> None:
        if GymWrapper is not object:
            super().__init__(env)
        self.env: PokemonEnv = env
        # Enable single-agent mode in the underlying environment
        setattr(self.env, "single_agent_mode", True)

        # 対戦相手としてランダム行動の ``MapleAgent`` を登録しておく
        opponent = MapleAgent(self.env)
        self.env.register_agent(opponent, "player_1")
        self._opponent = opponent

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying environment."""
        return getattr(self.env, name)

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space[self.env.agent_ids[0]]

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space[self.env.agent_ids[0]]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info

    def step(self, action: Any):
        """Advance the environment one step with an automatic opponent."""

        battle = self.env._current_battles.get(self.env.agent_ids[1])
        if battle is not None:
            if isinstance(action, str):
                opp_action = self._opponent.choose_team(
                    self.env.state_observer.observe(battle)
                )
            else:
                mask, _ = self.env.action_helper.get_available_actions_with_details(
                    battle
                )
                obs = self.env.state_observer.observe(battle)
                opp_action = self._opponent.select_action(obs, mask)
        else:
            opp_action = 0

        return self.env.step({"player_0": action, "player_1": opp_action})


def make_single_agent_env(**kwargs: Any) -> gym.Env:
    """Return :class:`PokemonEnv` wrapped for single-agent usage."""

    env = PokemonEnv(**kwargs)
    return SingleAgentCompatibilityWrapper(env)


if hasattr(gym, "register"):
    gym.register(
        "PokemonEnv-SA-v0",
        entry_point="src.env.wrappers:make_single_agent_env",
    )
