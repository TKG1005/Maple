# LSTM Learning Optimization Implementation Log

**Date**: 2025-07-10  
**Branch**: `feature/lstm-learning-optimization`  
**Commit**: `3bd25e8a6`

## Overview

Implementation of comprehensive LSTM learning optimization based on design document `docs/AI-design/M7/LSTM学習の適正化・バッチ学習方針.md`. This implementation addresses critical issues with LSTM sequence learning in reinforcement learning training.

## Problem Analysis

### Original Issues
1. **Episode Boundary Mixing**: LSTM hidden states were not properly reset between episodes, causing gradient mixing
2. **Sequence Structure Loss**: Traditional algorithms flattened batch data, breaking temporal dependencies
3. **Gradient Explosion**: Long LSTM sequences without proper clipping caused training instability
4. **Network Compatibility**: RLAgent only passed policy_net to algorithms, breaking sequence algorithms that need both networks

### Design Requirements
From the design document:
- エピソード境界での隠れ状態リセット (Hidden state reset at episode boundaries)
- エピソード単位・系列長単位のシーケンシャル学習 (Episode-based sequential learning)
- 勾配爆発への対策 (Gradient explosion countermeasures)
- 計算コストと実現性の検討 (Computational cost and feasibility)

## Implementation Details

### 1. Sequence-Based Algorithms

#### SequencePPOAlgorithm (`src/algorithms/sequence_ppo.py`)
```python
class SequencePPOAlgorithm(BaseAlgorithm):
    def __init__(self, bptt_length=0, grad_clip_norm=5.0):
        self.bptt_length = int(bptt_length)  # 0 = full episode
        self.grad_clip_norm = float(grad_clip_norm)
    
    def _process_sequence_step_by_step(self, policy_net, value_net, obs_sequence, device):
        """Process sequence maintaining LSTM hidden states"""
        policy_hidden = None
        value_hidden = None
        all_logits, all_values = [], []
        
        for t in range(seq_len):
            obs_t = obs_sequence[t:t+1]
            logits_t, policy_hidden = policy_net(obs_t, policy_hidden)
            value_t, value_hidden = value_net(obs_t, value_hidden)
            all_logits.append(logits_t.squeeze(0))
            all_values.append(value_t.squeeze())
        
        return torch.stack(all_logits), torch.stack(all_values)
```

**Key Features**:
- Step-by-step processing maintains temporal structure
- Hidden states preserved within episodes
- Configurable BPTT length for memory management
- Proper handling of both policy and value networks

#### SequenceReinforceAlgorithm
- Similar architecture but only needs policy network
- Maintains compatibility with existing REINFORCE interface
- Same BPTT and gradient clipping features

### 2. Enhanced Gradient Clipping

Added gradient clipping to all algorithms:

```python
# In PPOAlgorithm.update()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

# In SequencePPOAlgorithm.update() - configurable
torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=self.grad_clip_norm)
torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=self.grad_clip_norm)
```

### 3. RLAgent Integration

Enhanced `RLAgent.update()` to support sequence algorithms:

```python
def update(self, batch):
    algorithm_name = type(self.algorithm).__name__
    if algorithm_name in ["SequencePPOAlgorithm", "SequenceReinforceAlgorithm"]:
        # Pass both networks for sequence algorithms
        return self.algorithm.update((self.policy_net, self.value_net), self.optimizer, batch)
    else:
        # Standard algorithms only need policy network
        return self.algorithm.update(self.policy_net, self.optimizer, batch)
```

### 4. Episode Length Tracking

Modified episode data collection to include episode lengths:

```python
# In run_episode() and run_episode_with_opponent()
batch = {
    "observations": np.stack([t["obs"] for t in traj]),
    "actions": np.array([t["action"] for t in traj], dtype=np.int64),
    # ... other arrays ...
    "episode_lengths": np.array([len(traj)], dtype=np.int64),  # NEW
}
```

### 5. Automatic Algorithm Selection

Training script automatically selects sequence algorithms:

```python
# In train_selfplay.py
sequence_config = cfg.get("sequence_learning", {})
use_sequence_learning = sequence_config.get("enabled", False)

if use_sequence_learning and network_config.get("use_lstm", False):
    if algo_name == "ppo":
        algorithm = SequencePPOAlgorithm(
            bptt_length=bptt_length,
            grad_clip_norm=grad_clip_norm,
        )
    # ... else REINFORCE
else:
    # Standard algorithms
```

## Configuration Updates

### New Configuration Section
```yaml
# sequence_learning configuration
sequence_learning:
  enabled: true          # Enable sequence-based learning for LSTM
  bptt_length: 0         # 0=full episode, >0=truncated BPTT
  grad_clip_norm: 5.0    # Gradient clipping norm
```

### Template Updates
- **`config/train_config.yml`**: Testing template with sequence learning enabled, full episode BPTT
- **`config/train_config_long.yml`**: Production template with truncated BPTT (50 steps)

## Testing and Validation

### Comprehensive Test Suite (`test_sequence_learning.py`)

```python
def test_sequence_ppo_bptt():
    """Test sequence PPO with different BPTT lengths."""
    for bptt_length in [0, 10, 20]:
        algo = SequencePPOAlgorithm(bptt_length=bptt_length)
        loss = algo.update((policy_net, value_net), optimizer, batch)
        assert not np.isnan(loss)

def test_episode_boundary_handling():
    """Test that episode boundaries are properly handled."""
    episodes = [8, 12, 15, 20]  # Different episode lengths
    sequences = algo._split_episode_into_sequences(batch, episodes)
    # Verify no sequence crosses episode boundaries
```

**Test Results**: All tests pass, validating:
- ✅ Gradient clipping functionality
- ✅ Sequence PPO with different BPTT lengths
- ✅ Episode boundary handling
- ✅ Sequence REINFORCE algorithm
- ✅ Compatibility with standard algorithms

### Debug and Validation Process

1. **Identified Core Issue**: RLAgent passing single network instead of tuple
2. **Tensor Dimension Debugging**: Discovered value network returning wrong shapes
3. **MPS Device Issue**: Found PyTorch MPS LSTM gradient computation bug
4. **CPU Validation**: Confirmed working implementation on CPU device

## Performance Considerations

### Computational Impact
- **Memory Usage**: Step-by-step processing increases memory usage vs batch processing
- **Training Speed**: Slight slowdown due to sequential LSTM calls
- **BPTT Trade-off**: Full episode BPTT vs truncated for memory management

### Device Compatibility
- **CPU**: Full compatibility, recommended for LSTM training
- **CUDA**: Should work (not tested due to hardware limitations)
- **MPS (Apple Metal)**: Known PyTorch bug with LSTM gradients

## Known Issues and Limitations

### MPS Device Bug
```
Assertion failed: (shape4.size() >= 3), function _getLSTMGradKernelDAGObject, file GPURNNOps.mm, line 2417.
```
**Workaround**: Use CPU device for LSTM training
**Status**: PyTorch upstream issue

### Memory Scaling
- Full episode BPTT can consume significant memory for long episodes
- Truncated BPTT recommended for production training
- Configurable via `bptt_length` parameter

## Future Improvements

1. **Batch Sequence Processing**: Process multiple sequences in parallel
2. **Dynamic BPTT**: Adaptive BPTT length based on available memory
3. **MPS Compatibility**: Monitor PyTorch fixes for MPS LSTM support
4. **Performance Optimization**: Optimize step-by-step processing

## Files Modified

### Core Implementation
- `src/algorithms/sequence_ppo.py` (NEW)
- `src/algorithms/__init__.py`
- `src/algorithms/ppo.py`
- `src/algorithms/reinforce.py`
- `src/agents/RLAgent.py`

### Configuration
- `config/train_config.yml`
- `config/train_config_long.yml`

### Training Script
- `train_selfplay.py`

### Testing
- `test_sequence_learning.py` (NEW)

## Conclusion

The LSTM learning optimization implementation successfully addresses all requirements from the design document:

✅ **Episode Boundary Management**: Proper hidden state reset  
✅ **Sequential Learning**: Step-by-step processing maintains temporal structure  
✅ **Gradient Stability**: Comprehensive gradient clipping  
✅ **Computational Feasibility**: Configurable BPTT for memory management  

The implementation provides a solid foundation for LSTM-based reinforcement learning in Pokemon battles while maintaining compatibility with existing algorithms and training infrastructure.