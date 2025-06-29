import subprocess
import sys


def test_train_selfplay_cli():
    script = '''
import sys, types, runpy
class DummyArray(list):
    def __sub__(self, other):
        return DummyArray([a - b for a, b in zip(self, other)])

def default_rng(seed=None):
    class RNG:
        def choice(self, n, p=None):
            return 0
    return RNG()
np = types.ModuleType("numpy")
np.float32 = 'float32'
np.int8 = 'int8'
np.int64 = 'int64'
np.ndarray = DummyArray
np.random = types.SimpleNamespace(default_rng=default_rng)
np.zeros = lambda shape, dtype=None: DummyArray([0] * shape)
np.array = lambda data, dtype=None: DummyArray(list(data))
np.stack = lambda seq, axis=0: DummyArray([item for sub in seq for item in (sub if isinstance(sub, list) else [sub])])
np.log = lambda x: 0.0
np.array2string = lambda arr, precision=None: str(list(arr))
sys.modules['numpy'] = np

yaml = types.ModuleType('yaml')
yaml.safe_load = lambda f: {}
sys.modules['yaml'] = yaml

torch = types.ModuleType('torch')
torch.float32 = 'float32'
torch.nn = types.ModuleType('torch.nn')
torch.optim = types.ModuleType('torch.optim')
torch.optim.Adam = lambda params, lr=None: object()
torch.save = lambda *a, **k: None
torch.as_tensor = lambda x, dtype=None: x
sys.modules['torch'] = torch
sys.modules['torch.nn'] = torch.nn
sys.modules['torch.optim'] = torch.optim
sys.modules['torch.utils'] = types.ModuleType('torch.utils')
tb = types.ModuleType('torch.utils.tensorboard')
tb.SummaryWriter = lambda *a, **k: types.SimpleNamespace(add_scalar=lambda *a, **k: None, close=lambda: None)
sys.modules['torch.utils.tensorboard'] = tb

src = types.ModuleType('src')
sys.modules['src'] = src
env = types.ModuleType('src.env')
sys.modules['src.env'] = env
penv = types.ModuleType('src.env.pokemon_env')
class DummyEnv:
    def __init__(self, *a, **k):
        self.observation_space = {'player_0': DummyArray([0]), 'player_1': DummyArray([0])}
        self.action_space = {'player_0': DummyArray([0]), 'player_1': DummyArray([0])}
        self.agent_ids = ('player_0', 'player_1')
        self.rng = default_rng()
    def register_agent(self, agent, id):
        pass
    def reset(self, *a, return_masks=True, **k):
        return (
            {'player_0': DummyArray([0]), 'player_1': DummyArray([0])},
            {'request_teampreview': False},
            (DummyArray([1]), DummyArray([1])),
        )
    def get_action_mask(self, id, with_details=False):
        return DummyArray([1]), {0: ('move', 0, False)}
    def step(self, actions, return_masks=True):
        obs = {'player_0': DummyArray([0]), 'player_1': DummyArray([0])}
        rewards = {'player_0': 0.0, 'player_1': 0.0}
        terms = {'player_0': True, 'player_1': True}
        truncs = {'player_0': False, 'player_1': False}
        masks = (DummyArray([1]), DummyArray([1]))
        return obs, rewards, terms, truncs, {}, masks
    def close(self):
        pass
penv.PokemonEnv = DummyEnv
sys.modules['src.env.pokemon_env'] = penv

state = types.ModuleType('src.state.state_observer')
state.StateObserver = lambda path: types.SimpleNamespace(get_observation_dimension=lambda: 1, observe=lambda b: DummyArray([0]))
sys.modules['src.state.state_observer'] = state

action = types.ModuleType('src.action')
action.action_helper = types.SimpleNamespace(
    action_index_to_order=lambda self, battle, idx: types.SimpleNamespace(message=''),
    get_available_actions=lambda battle: (DummyArray([1]), {0: ('move', 0, False)}),
    get_available_actions_with_details=lambda battle: (DummyArray([1]), {0: {'type': 'move'}}),
)
sys.modules['src.action'] = action

agents = types.ModuleType('src.agents')
class PolicyNetwork:
    def __init__(self, *a, **k):
        pass
    def parameters(self):
        return []
    def __call__(self, obs):
        return DummyArray([0.0])
class ValueNetwork:
    def __init__(self, *a, **k):
        pass
    def parameters(self):
        return []
    def __call__(self, obs):
        return types.SimpleNamespace(item=lambda: 0.0)
class RLAgent:
    def __init__(self, env, policy_net, value_net, optimizer, algorithm=None):
        pass
    def select_action(self, obs, mask):
        return DummyArray([1.0])
    def update(self, batch):
        return 0.0
agents.PolicyNetwork = PolicyNetwork
agents.ValueNetwork = ValueNetwork
agents.RLAgent = RLAgent
sys.modules['src.agents'] = agents

alg = types.ModuleType('src.algorithms')
alg.PPOAlgorithm = lambda *a, **k: types.SimpleNamespace()
alg.ReinforceAlgorithm = lambda *a, **k: types.SimpleNamespace()
alg.compute_gae = lambda rewards, values, gamma=0.99, lam=1.0: [0.0] * len(rewards)
sys.modules['src.algorithms'] = alg

sys.argv = ['train_selfplay.py', '--episodes', '1', '--clip', '0.1', '--ppo-epochs', '2']
runpy.run_path('train_selfplay.py', run_name='__main__')
'''
    result = subprocess.run([sys.executable, '-'], input=script, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    out = result.stdout + result.stderr
    assert "'clip': 0.1" in out
    assert "'ppo_epochs': 2" in out
