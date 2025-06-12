import numpy as np

from src.env.pokemon_env import PokemonEnv
from src.agents.MapleAgent import MapleAgent


class DummyObserver:
    def get_observation_dimension(self) -> int:
        return 1

    def observe(self, battle):
        return np.zeros(1, dtype=np.float32)


class DummyActionHelper:
    def get_available_actions_with_details(self, battle):
        return [1] + [0] * 9, {}

    def action_index_to_order(self, env_player, battle, idx):
        return None


class DummyOpponent:
    def reset_battles(self):
        pass


# Patch PokemonEnv methods to avoid external dependencies

def _dummy_reset(self, *args, **kwargs):
    self._turn = 0
    obs = {agent_id: np.zeros(1, dtype=np.float32) for agent_id in self.agent_ids}
    return obs


def _dummy_step(self, action_dict):
    self._turn += 1
    obs = {agent_id: np.zeros(1, dtype=np.float32) for agent_id in self.agent_ids}
    reward = {agent_id: 0.0 for agent_id in self.agent_ids}
    terminated = self._turn >= 3
    if terminated:
        reward = {self.agent_ids[0]: 1.0, self.agent_ids[1]: -1.0}
    term = {agent_id: terminated for agent_id in self.agent_ids}
    trunc = {agent_id: False for agent_id in self.agent_ids}
    info = {agent_id: {} for agent_id in self.agent_ids}
    return obs, reward, term, trunc, info


PokemonEnv.reset = _dummy_reset  # type: ignore
PokemonEnv.step = _dummy_step  # type: ignore
PokemonEnv.close = lambda self: None  # type: ignore


class RandomAgent(MapleAgent):
    def select_action(self, observation, mask):
        return 0


if __name__ == "__main__":
    env = PokemonEnv(
        opponent_player=DummyOpponent(),
        state_observer=DummyObserver(),
        action_helper=DummyActionHelper(),
    )
    agent0 = RandomAgent(env)
    agent1 = RandomAgent(env)

    obs = env.reset()
    done = False
    total_reward = {agent_id: 0.0 for agent_id in env.agent_ids}
    while not done:
        actions = {agent_id: 0 for agent_id in env.agent_ids}
        obs, reward, term, trunc, _ = env.step(actions)
        for agent_id in env.agent_ids:
            total_reward[agent_id] += reward[agent_id]
        done = any(term.values()) or any(trunc.values())

    print("Episode finished. Rewards:")
    for agent_id in env.agent_ids:
        print(f"  {agent_id}: {total_reward[agent_id]}")
