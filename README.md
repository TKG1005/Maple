# Maple - Pokemon Reinforcement Learning Framework

Maple is a Pokemon reinforcement learning framework built on top of `poke-env` and Pokemon Showdown. It implements multi-agent self-play training for Pokemon battles using deep reinforcement learning algorithms (PPO, REINFORCE).

## Changelog

### 2025-07-10 - LSTM Learning Optimization and Sequence-Based Training

#### 🎯 **Major Features**
- **Sequence-Based Algorithms**: New `SequencePPOAlgorithm` and `SequenceReinforceAlgorithm` for proper LSTM sequence learning
- **Configurable BPTT**: Support for full episode or truncated backpropagation through time
- **Enhanced Gradient Clipping**: Gradient clipping added to all algorithms for training stability
- **Automatic Algorithm Selection**: Smart selection of sequence algorithms for LSTM networks

#### 🔧 **Implementation Details**
- **Step-by-step Processing**: Maintains LSTM hidden states across timesteps within episodes
- **Episode Boundary Management**: Proper hidden state reset at episode boundaries
- **Enhanced RLAgent**: Automatic detection and handling of sequence algorithms
- **Episode Length Tracking**: Added episode_lengths to batch data for sequence splitting

#### ⚙️ **Configuration**
- **New Config Section**: `sequence_learning` configuration in YAML files
- **Template Updates**: Updated `train_config.yml` and `train_config_long.yml`
- **Device Compatibility**: CPU device recommended for LSTM training (MPS has known issues)

#### 🧪 **Testing**
- **Comprehensive Test Suite**: `test_sequence_learning.py` validates all sequence learning features
- **Debug Capabilities**: Enhanced debugging tools for LSTM sequence processing
- **Algorithm Comparison**: Tests verify compatibility between standard and sequence algorithms

#### 📁 **Configuration Example**
```yaml
sequence_learning:
  enabled: true          # Enable sequence-based learning for LSTM
  bptt_length: 0         # 0=full episode, >0=truncated BPTT  
  grad_clip_norm: 5.0    # Gradient clipping norm
```

### 2025-07-09 - LSTM Conflict Resolution and GPU Support

#### 🔄 **LSTM Hidden State Management**
- **Stateless Networks**: Refactored LSTM networks to return hidden states instead of storing them
- **Agent-Level State Management**: RLAgent now manages hidden states per agent instance
- **Thread Safety**: LSTM networks now safe for parallel execution

#### 🚀 **GPU Acceleration Support**
- **Multi-Platform Support**: CUDA, Apple MPS, and CPU fallback
- **Automatic Device Detection**: Intelligent device selection with graceful fallback
- **Memory Management**: Proper GPU memory handling and cleanup

#### 🎮 **Training Enhancements**
- **Self-Play Architecture**: Single-model convergence with frozen opponent system
- **Reward Normalization**: Comprehensive reward normalization for stable training
- **Configuration System**: YAML-based configuration management

### Previous Updates
- **Value Network Hidden State Management**: Enhanced LSTM value network processing
- **Win Rate-Based Opponent Updates**: Intelligent opponent update system
- **Network Forward Method Compatibility**: Fixed compatibility between basic and enhanced networks

## Usage

### Quick Start
```bash
# Basic training with sequence learning
python train_selfplay.py --config config/train_config.yml

# Long-term training with truncated BPTT
python train_selfplay.py --config config/train_config_long.yml

# CPU training (recommended for LSTM)
python train_selfplay.py --config config/train_config.yml --device cpu
```

### Evaluation
```bash
# Evaluate trained model
python evaluate_rl.py --model checkpoints/checkpoint_ep14000.pt --opponent random --n 10
```

## Requirements

- Python 3.9+
- PyTorch 1.12+
- Pokemon Showdown server
- See `requirements.txt` for full dependencies

## Documentation

- `CLAUDE.md`: Comprehensive project documentation and implementation details
- `docs/`: Design documents and implementation logs
- `config/`: Configuration templates and examples

## License

This project is licensed under the MIT License.