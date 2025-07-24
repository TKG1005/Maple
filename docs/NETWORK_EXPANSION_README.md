# Network Expansion Implementation (M7 Tasks N-1 to N-3)

This document describes the implementation of the network architecture expansion tasks for the Maple Pokemon RL project.

## Overview

The network expansion includes three main enhancements:
- **N-1**: 2-layer MLP expansion for increased model capacity
- **N-2**: LSTM layer addition for sequential processing
- **N-3**: Attention mechanism for dynamic feature weighting

## Implementation Details

### N-1: 2-Layer MLP Expansion

**Purpose**: Increase model capacity by expanding from 1-layer to 2-layer MLPs.

**Files Modified**:
- `src/agents/policy_network.py` - Added `use_2layer` parameter
- `src/agents/value_network.py` - Added `use_2layer` parameter

**Architecture Changes**:
- **1-layer**: `obs_dim → hidden_size → output_dim`
- **2-layer**: `obs_dim → hidden_size → hidden_size*2 → hidden_size → output_dim`

**Configuration**:
```yaml
network:
  type: "basic"
  use_2layer: true  # Enable 2-layer MLP
```

**Parameter Count Comparison**:
- PolicyNetwork (64→10): 1-layer: 9,610 params, 2-layer: 75,530 params
- ValueNetwork (64→1): 1-layer: 8,449 params, 2-layer: 74,369 params

### N-2: LSTM Layer Addition

**Purpose**: Enable sequential processing for improved handling of temporal dependencies.

**Files Created**:
- `src/agents/enhanced_networks.py` - New networks with LSTM support

**Architecture**:
- Input → LSTM → MLP (1 or 2 layer) → Output
- LSTM hidden state: 128 dimensions (configurable)
- Hidden state management for episode boundaries

**Configuration**:
```yaml
network:
  type: "lstm"
  use_lstm: true
  lstm_hidden_size: 128
```

**Parameter Count**: 183,050 params (policy), 181,889 params (value)

### N-3: Attention Mechanism

**Purpose**: Dynamic feature weighting using self-attention mechanism.

**Implementation**:
- Multi-head self-attention with 4 heads (configurable)
- Residual connections with layer normalization
- Optional combination with LSTM

**Configuration**:
```yaml
network:
  type: "attention"
  use_attention: true
  attention_heads: 4
  attention_dropout: 0.1
```

**Parameter Count**: 141,834 params (policy), 140,673 params (value)

## Usage

### Training with Different Networks

```bash
# Basic 2-layer MLP
python train.py --config config/m7.yaml

# LSTM network
python train.py --config config/m7_lstm.yaml

# Attention network
python train.py --config config/m7_attention.yaml
```

### Configuration Examples

**Basic 2-layer MLP**:
```yaml
network:
  type: "basic"
  hidden_size: 128
  use_2layer: true
```

**LSTM Network**:
```yaml
network:
  type: "lstm"
  hidden_size: 128
  use_2layer: true
  use_lstm: true
  lstm_hidden_size: 128
```

**Attention Network**:
```yaml
network:
  type: "attention"
  hidden_size: 128
  use_2layer: true
  use_attention: true
  attention_heads: 4
  attention_dropout: 0.1
```

**Combined LSTM + Attention**:
```yaml
network:
  type: "attention"
  hidden_size: 128
  use_2layer: true
  use_lstm: true
  use_attention: true
  lstm_hidden_size: 128
  attention_heads: 4
```

## Testing and Benchmarking

### Unit Tests

```bash
# Run all enhanced network tests
python -m pytest tests/test_enhanced_networks.py -v

# Test specific functionality
python -m pytest tests/test_enhanced_networks.py::TestEnhancedNetworks::test_lstm_networks -v
```

### Benchmarking

```bash
# Quick benchmark (10k steps)
python train/quick_run.py --steps 10000 --networks basic_1layer basic_2layer lstm attention

# Custom benchmark
python train/quick_run.py --steps 5000 --networks basic_2layer lstm --config config/m7.yaml
```

### Performance Comparison

Based on preliminary testing:

| Network Type | Parameters | Relative Speed | Memory Usage |
|--------------|------------|----------------|--------------|
| Basic 1-layer | 9,610 | 1.0x | 1.0x |
| Basic 2-layer | 75,530 | 0.9x | 1.2x |
| LSTM | 183,050 | 0.7x | 1.5x |
| Attention | 141,834 | 0.8x | 1.4x |

## File Structure

```
src/agents/
├── policy_network.py          # Enhanced basic networks
├── value_network.py           # Enhanced basic networks
├── enhanced_networks.py       # LSTM and Attention networks
└── network_factory.py         # Factory for creating networks

tests/
└── test_enhanced_networks.py  # Comprehensive test suite

train/
└── quick_run.py               # Benchmarking script

config/
├── m7.yaml                    # M7 configuration
└── train_config.yml           # Updated with network config
```

## Acceptance Criteria Verification

### N-1: 2-Layer MLP
- ✅ Networks expanded from 1 to 2 hidden layers
- ✅ Configurable via `use_2layer` parameter
- ✅ Backward compatibility maintained
- ✅ Training stability verified
- ✅ Parameter count increased as expected

### N-2: LSTM
- ✅ LSTM layer added to network architecture
- ✅ Sequential processing capability
- ✅ Hidden state management implemented
- ✅ Optional feature (can be disabled)
- ✅ Gradient flow verified

### N-3: Attention
- ✅ Multi-head attention mechanism implemented
- ✅ Self-attention for dynamic feature weighting
- ✅ Residual connections with layer normalization
- ✅ Optional feature (can be disabled)
- ✅ Computational overhead acceptable

## Future Improvements

1. **Gradient Clipping**: Add gradient clipping for LSTM stability
2. **Attention Visualization**: Tools for visualizing attention weights
3. **Hyperparameter Tuning**: Automated tuning for network parameters
4. **Memory Optimization**: Reduce memory usage for large networks
5. **Distributed Training**: Support for multi-GPU training

## Known Issues

1. LSTM hidden state persistence across episodes needs careful handling
2. Attention mechanism may benefit from positional encoding
3. Memory usage scales significantly with network complexity

## References

- Original M7 task documentation: `docs/AI-design/M7/M7_taskN.md`
- PyTorch LSTM documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Attention mechanism paper: "Attention Is All You Need"