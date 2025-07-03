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
- **Reward System**: Modular reward components (HP delta, knockouts, turn penalties, fail/immune actions)
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
# Evaluate trained model
python evaluate_rl.py

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
- `HPDeltaReward`: Rewards based on HP changes
- `KnockoutReward`: Rewards for fainting opposing Pokemon
- `TurnPenaltyReward`: Penalizes long battles
- `FailAndImmuneReward`: Penalizes failed and immune moves (default penalty: -0.02)

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