"""Environment wrappers for compatibility."""

from __future__ import annotations

from typing import Any


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
        from src.agents.MapleAgent import MapleAgent
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
        data = self.env.reset(seed=seed, options=options)
        observation = data["obs"][self.env.agent_ids[0]]
        info = data["info"]
        return observation, info

    def step(self, action: Any):
        """Advance the environment one step with an automatic opponent."""

        opp_id = self.env.agent_ids[1]
        opp_action = 0
        if self.env._need_action.get(opp_id, False):
            battle = self.env.get_current_battle(opp_id)
            if battle is not None:
                if isinstance(action, str):
                    obs = self.env.state_observer.observe(battle)
                    opp_action = self._opponent.choose_team(obs)
                else:
                    mask, mapping = self.env.action_helper.get_available_actions(battle)
                    # マスク生成に使用したマッピングを環境にも共有しておく
                    self.env._action_mappings[opp_id] = mapping
                    obs = self.env.state_observer.observe(battle)
                    opp_action = self._opponent.select_action(obs, mask)

        result = self.env.step({"player_0": action, "player_1": opp_action})
        obs = result["obs"][self.env.agent_ids[0]]
        reward = result["rewards"][self.env.agent_ids[0]]
        done = result["terminated"][self.env.agent_ids[0]] or result["truncated"][self.env.agent_ids[0]]
        info = result["info"][self.env.agent_ids[0]]
        mask, mapping = self.env.get_action_mask(self.env.agent_ids[0])
        self.env._action_mappings[self.env.agent_ids[0]] = mapping
        return obs, mask, reward, done, info


def make_single_agent_env(**kwargs: Any) -> gym.Env:
    """Return :class:`PokemonEnv` wrapped for single-agent usage."""

    env = PokemonEnv(**kwargs)
    return SingleAgentCompatibilityWrapper(env)


if hasattr(gym, "register"):
    gym.register(
        "PokemonEnv-SA-v0",
        entry_point="src.env.wrappers:make_single_agent_env",
    )
