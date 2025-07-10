# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Maple is a Pokemon reinforcement learning framework built on top of `poke-env` and Pokemon Showdown. It implements multi-agent self-play training for Pokemon battles using deep reinforcement learning algorithms (PPO, REINFORCE).

## Core Architecture

### Main Components

- **PokemonEnv**: Multi-agent environment that interfaces with Pokemon Showdown server via WebSocket
- **MapleAgent**: Base agent class for battle decision making
- **EnvPlayer**: Bridge between poke-env's Player class and PokemonEnv
- **StateObserver**: Converts battle state into numerical feature vectors for ML models
- **Reward System**: Modular reward components (knockouts, turn penalties, fail/immune actions, Pokemon count difference)
- **Algorithms**: PPO and REINFORCE implementations with GAE for policy gradient methods

### Key Architecture Patterns

- **Async-Sync Bridge**: Uses `asyncio.run_coroutine_threadsafe()` to integrate poke-env's async WebSocket handling with synchronous RL training loops
- **Multi-Agent Dict Interface**: Environment returns observations, actions, rewards as dictionaries keyed by player ID
- **Modular Rewards**: CompositeReward class combines multiple reward components with configurable weights
- **Action Masking**: Valid actions are computed dynamically and passed to agents to prevent invalid moves

## Development Commands

### Training
```bash
# Basic self-play training with default config
python train_selfplay.py

# PPO training with custom parameters
python train_selfplay.py --algo ppo --episodes 100 --ppo-epochs 4 --clip 0.2 --save model.pt

# Training with specific reward configuration
python train_selfplay.py --reward composite --reward-config config/reward.yaml

# Training with TensorBoard logging
python train_selfplay.py --tensorboard

# Resume training from a checkpoint
python train_selfplay.py --load-model checkpoints/checkpoint_ep14000.pt --episodes 100

# Training with random teams
python train_selfplay.py --team random --teams-dir config/teams --episodes 50

# Training against specific opponents
python train_selfplay.py --opponent rule --episodes 100
python train_selfplay.py --opponent-mix "random:0.3,max:0.3,self:0.4" --episodes 100

# Advanced training with multiple options
python train_selfplay.py --load-model checkpoints/checkpoint_ep5000.pt --team random --opponent-mix "rule:0.5,self:0.5" --episodes 100 --checkpoint-interval 10
```

### Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_composite_reward.py
pytest -m slow  # Long-running tests
```

### Evaluation
```bash
# Basic model evaluation against random opponent
python evaluate_rl.py --model checkpoints/checkpoint_ep14000.pt --opponent random --n 10

# Evaluate with random teams for varied battles
python evaluate_rl.py --model checkpoints/checkpoint_ep14000.pt --opponent rule --team random --n 5

# Head-to-head model comparison
python evaluate_rl.py --models checkpoints/model_a.pt checkpoints/model_b.pt --n 10

# Evaluation with replay saving and custom team directory
python evaluate_rl.py --model checkpoints/checkpoint_ep14000.pt --opponent max --team random --teams-dir config/teams --replay-dir my_replays --n 3

# Plot training results comparison
python plot_compare.py
```

## Configuration Files

- `config/train_config.yml`: Main training hyperparameters (learning rate, batch size, algorithms)
- `config/reward.yaml`: Reward component weights and enablement flags
- `config/env_config.yml`: Environment settings
- `config/action_map.yml`: Action space configuration
- `config/state_spec.yml`: State observation specification

## Key Implementation Details

### Environment Synchronization
The system uses a complex synchronization mechanism between async poke-env and sync RL training:
1. EnvPlayer monitors `battle.last_request` changes to detect server updates
2. Battle objects are passed to PokemonEnv only after request ID changes
3. Action queues coordinate between RL agents and poke-env players

### State Representation
StateObserver creates feature vectors from battle state including:
- Pokemon stats, moves, types
- Team composition and status
- Type effectiveness calculations
- Battle field conditions

### Reward Engineering
Multiple reward components can be combined:
- `KnockoutReward`: Rewards for fainting opposing Pokemon
- `TurnPenaltyReward`: Penalizes long battles
- `FailAndImmuneReward`: Penalizes failed and immune moves (default penalty: -0.02)
- `PokemonCountReward`: End-game rewards based on remaining Pokemon difference (1 diff: 0 pts, 2 diff: ±2 pts, 3+ diff: ±5 pts)

## Testing Strategy

The project uses pytest with custom markers:
- Regular unit tests for reward functions and utilities
- `@pytest.mark.slow` for integration tests involving actual battles
- Separate test directories: `test/` for integration, `tests/` for unit tests

## Pokemon Showdown Integration

The project includes a full Pokemon Showdown server in `pokemon-showdown/` directory. The environment connects to `localhost:8000` by default for battle simulation.

## Model Architecture

- **Policy Network**: Maps state observations to action probabilities
- **Value Network**: Estimates state values for advantage calculation
- **Shared Features**: Both networks can share initial layers for efficiency
- **Action Masking**: Output layer respects valid action constraints

## Common Development Patterns

When modifying the codebase:
1. New reward components should inherit from `RewardBase` and be registered in `CompositeReward.DEFAULT_REWARDS`
2. Algorithm implementations should inherit from `BaseAlgorithm`
3. State features should be documented in `state_spec.yml`
4. Configuration changes should update corresponding YAML files
5. Integration tests should be marked with `@pytest.mark.slow`

### FailAndImmuneReward Implementation Details
The `FailAndImmuneReward` class provides penalties for invalid actions:
- Monitors `battle.last_fail_action` and `battle.last_immune_action` flags
- Default penalty: -0.02 (configurable via constructor)
- Stateless design - no internal state to reset
- Enabled via `config/reward.yaml` with `fail_immune.enabled: true`

### Training Resume Functionality (train_selfplay.py)
The `--load-model` option enables resuming training from checkpoints:
- Supports both new format (`{"policy": ..., "value": ...}`) and legacy format (single state dict)
- Automatically extracts episode number from filenames like `checkpoint_ep14000.pt`
- Continues episode numbering from the extracted number
- Example: `python train_selfplay.py --load-model checkpoints/checkpoint_ep5000.pt --episodes 100`

### Random Team System
The `--team random` option enables varied training and evaluation:
- Each player independently selects a random team from `config/teams/` directory
- Teams are selected per battle, ensuring variety across episodes
- Custom team directory can be specified with `--teams-dir`
- Supports both training (`train_selfplay.py`) and evaluation (`evaluate_rl.py`)
- Team files must be in Pokemon Showdown format

### Enhanced Player Naming (evaluate_rl.py)
Player names are automatically made unique to prevent NameTaken errors:
- Uses timestamp and random numbers to ensure uniqueness
- Names are truncated to 18 characters (Pokemon Showdown limit)
- Model names and agent types are preserved for replay identification
- Format: `ModelName_UniqueID` (e.g., `checkpoint_3418161`, `RuleBasedP_3418161`)

### Opponent Mixing System
The `--opponent-mix` option allows training against multiple opponent types:
- Format: `"type1:ratio1,type2:ratio2,type3:ratio3"`
- Example: `"random:0.3,max:0.3,self:0.4"` (30% random, 30% max damage, 40% self-play)
- Supports `random`, `max`, `rule`, and `self` opponent types
- Randomly selects opponent type based on specified ratios for each episode

### Self-Play Architecture (Updated 2025-07-09)
The self-play system implements a **single-model convergence approach**:
- **Main Agent**: Learns and updates through training, uses optimizer
- **Opponent Agent**: Uses frozen copy of main agent's current weights
- **Weight Copying**: Opponent networks are refreshed with main agent weights each episode
- **Network Freezing**: Opponent networks have `requires_grad=False` and no optimizer
- **Final Output**: Single trained model representing the learned policy
- **Learning Dynamics**: Main agent progressively improves by playing against its own current strength

### Reward Normalization System (New 2025-07-09)
Comprehensive reward normalization for stable training:
- **RewardNormalizer**: Running mean/std normalization with Welford's algorithm
- **WindowedRewardNormalizer**: Sliding window-based normalization (alternative)
- **Per-Agent Normalization**: Independent normalizers for each agent
- **Integration**: Automatic normalization in `PokemonEnv._calc_reward()`
- **Configuration**: Enable/disable via `normalize_rewards` parameter
- **Statistics**: Access normalization stats via `env.get_reward_normalization_stats()`

### LSTM Hidden State Management Fix (New 2025-07-10)
Fixed critical issue where LSTM networks were not maintaining sequential learning capability:

**Problem**: `RLAgent.select_action()` did not pass hidden states to LSTM networks, causing them to reinitialize on every action selection. This defeated the purpose of LSTM's sequential learning.

**Solution**: 
- Modified `RLAgent` to detect LSTM/Attention networks with `has_hidden_states` flag
- Added `reset_hidden_states()` method to reset hidden states at episode boundaries
- Updated `select_action()` to pass hidden states to LSTM networks
- Added tensor dimension handling for batch processing
- Modified training loops to call `reset_hidden_states()` at episode start

**Implementation Details**:
```python
# In RLAgent.__init__()
self.has_hidden_states = hasattr(policy_net, 'hidden_state') and (value_net is None or hasattr(value_net, 'hidden_state'))

# In RLAgent.select_action()
if self.has_hidden_states:
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    logits = self.policy_net(obs_tensor, self.policy_net.hidden_state)
    if logits.dim() == 2 and logits.size(0) == 1:
        logits = logits.squeeze(0)

# In run_episode() and run_episode_with_opponent()
agent.reset_hidden_states()  # Called at episode start
```

**Testing**: Added comprehensive tests (`test_lstm_sequential_learning.py`) to verify:
- Hidden states change across action selections
- Hidden states are reset properly at episode boundaries
- Sequential learning produces different outputs for different histories
- LSTM networks behave differently from basic networks

### Configuration File System (New 2025-07-10)
Implemented comprehensive YAML-based configuration management to simplify training execution:

**Problem**: Training required long command lines with many parameters, making it difficult to manage different training scenarios and reproduce experiments.

**Solution**: Created YAML configuration system with two main templates:
- `config/train_config.yml`: Testing and short-term training (10 episodes, mixed opponents, LSTM network)
- `config/train_config_long.yml`: Long-term training (1000 episodes, self-play, attention network)

**Features**:
- All training parameters configurable via YAML
- Command line arguments override config file values
- Network architecture configuration in config files
- Detailed configuration logging
- Parameter validation and type conversion

**Usage Examples**:
```bash
# Testing and short-term training
python train_selfplay.py --config config/train_config.yml

# Long-term production training
python train_selfplay.py --config config/train_config_long.yml

# Config file with parameter override
python train_selfplay.py --config config/train_config.yml --episodes 20 --lr 0.001
```

### Win Rate-Based Opponent Update System (New 2025-07-10)
Implemented intelligent opponent update system to reduce excessive network copying in self-play:

**Problem**: Traditional self-play copied opponent networks every episode, leading to inefficient learning due to constantly changing opponents.

**Solution**: Conditional opponent updates based on win rate threshold:
- Monitor recent battle results using win_loss reward component
- Update opponent network only when win rate exceeds threshold (default 60%)
- Maintain opponent snapshots for consistent training
- Configurable win rate threshold and monitoring window

**Implementation**:
```python
def should_update_opponent(episode_num, battle_results, window_size, threshold):
    """Check if opponent should be updated based on recent win rate."""
    if len(battle_results) < window_size:
        return False
    recent_results = battle_results[-window_size:]
    wins = sum(1 for result in recent_results if result == 1)
    win_rate = wins / len(recent_results)
    return win_rate >= threshold
```

**Configuration**:
```yaml
# Self-play win rate based opponent update
win_rate_threshold: 0.6  # Win rate threshold for updating opponent
win_rate_window: 50      # Number of recent battles to track
```

**Benefits**:
- Reduced network copying frequency
- More stable learning against consistent opponents
- Improved training efficiency
- Configurable thresholds for different scenarios

### Network Forward Method Compatibility Fix (New 2025-07-10)
Fixed compatibility issue between basic networks and enhanced LSTM/Attention networks:

**Problem**: Basic networks (PolicyNetwork, ValueNetwork) only accept one argument in forward(), while enhanced networks (LSTM, Attention) accept optional hidden state parameters.

**Solution**: Added conditional forward method calls based on network capabilities:
```python
# Call value network with hidden state only if supported
if hasattr(value_net, 'hidden_state'):
    val0_tensor = value_net(obs0_tensor, value_net.hidden_state)
else:
    val0_tensor = value_net(obs0_tensor)
```

This ensures compatibility across all network types while maintaining optimal performance for each architecture.

## Project-Specific Rules

### Code Style and Conventions
- Use `from __future__ import annotations` for type hints
- All new agent classes must inherit from `MapleAgent` base class
- Use `MapleAgent.select_action(observation, action_mask)` interface for agents
- Implement both `select_action()` and `act()` methods for agent compatibility
- Import order: stdlib → third-party → local imports with `# noqa: E402` for late imports

### Agent Development Rules
- All agents must handle empty action masks gracefully
- Use `self.env.rng` for reproducible random number generation
- Agent constructors should accept `env` as first parameter
- Register agents with environment using `env.register_agent(agent, player_id)`
- Battle state access should use `self._get_current_battle()` pattern
- **RLAgent Optimizer**: Can accept `None` optimizer for frozen agents (self-play opponents)
- **Algorithm Updates**: All algorithms must handle `None` optimizer gracefully

### Testing Requirements
- Mark slow integration tests with `@pytest.mark.slow`
- Unit tests for reward functions must test edge cases (empty battles, missing attributes)
- Agent tests should verify both valid action selection and error handling
- Configuration changes require corresponding test updates

### Documentation Updates
- Update `README.md` changelog for new features
- Mark completed tasks in `docs/TODO_M7.md`
- Update this `CLAUDE.md` with new implementation details
- Include usage examples for new command-line options

### Prohibited Patterns
- Never use `poke_env.Player` directly - always use `MapleAgent` base class
- Avoid hardcoded paths - use relative paths from project root
- Don't create agents without proper environment registration
- Never ignore action masks - always respect valid action constraints
- **Self-Play**: Never create two learning agents in self-play (use frozen opponent instead)
- **Reward Normalization**: Don't reset normalizers between episodes (maintains running stats)

## Recent Updates (2025-07-09)

### Critical Bug Fixes
- **Self-Play Network Sharing**: Fixed issue where both agents shared the same network weights
- **Learning Rate Optimization**: Reduced from 0.002 to 0.0005 for more stable training
- **Reward Normalization**: Implemented comprehensive normalization system for training stability

### Architecture Improvements
- **Single-Model Convergence**: Self-play now produces a single final model instead of two competing models
- **Frozen Opponent System**: Opponent agents use current main agent weights but don't learn
- **Algorithm Flexibility**: All RL algorithms now support both learning and non-learning modes

### Configuration Updates
- **Training Config**: Updated `config/train_config.yml` with optimized hyperparameters
- **Reward Weights**: Adjusted `config/reward.yaml` based on performance analysis
- **Normalization**: New `normalize_rewards` parameter in environment initialization