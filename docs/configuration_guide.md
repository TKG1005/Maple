# Configuration Guide - LSTM Learning Optimization

This guide covers the new configuration options for LSTM learning optimization and sequence-based training.

## Sequence Learning Configuration

### Basic Configuration

Add the following section to your YAML configuration file:

```yaml
sequence_learning:
  enabled: true          # Enable sequence-based learning for LSTM networks
  bptt_length: 0         # Backpropagation Through Time length (0 = full episode)
  grad_clip_norm: 5.0    # Gradient clipping norm for stability
```

### Parameters

#### `enabled` (boolean, default: `false`)
- **Purpose**: Enables sequence-based algorithms for LSTM networks
- **Effect**: When `true` and `use_lstm: true`, automatically uses `SequencePPOAlgorithm` or `SequenceReinforceAlgorithm`
- **When to use**: Always enable for LSTM networks to ensure proper sequence learning

#### `bptt_length` (integer, default: `0`)
- **Purpose**: Controls the length of backpropagation through time
- **Values**:
  - `0`: Full episode BPTT (recommended for short episodes)
  - `N > 0`: Truncated BPTT with N timesteps (recommended for long episodes)
- **Memory Impact**: Larger values use more memory but provide better gradient information
- **Recommended Values**:
  - Testing: `0` (full episode)
  - Short episodes (<50 steps): `0`
  - Long episodes (>100 steps): `20-50`
  - Production training: `50`

#### `grad_clip_norm` (float, default: `5.0`)
- **Purpose**: Gradient clipping norm to prevent gradient explosion
- **Values**: Typically between `1.0` and `10.0`
- **Effect**: Higher values allow larger gradients but may cause instability
- **Recommended Values**:
  - Simple models: `5.0`
  - Complex models: `10.0`
  - Stable training: `2.0-3.0`

## Configuration Templates

### Testing Configuration (`config/train_config.yml`)

```yaml
# Sequence learning configuration - Testing
sequence_learning:
  enabled: true
  bptt_length: 0      # Full episode for testing
  grad_clip_norm: 5.0

# Network configuration
network:
  type: "lstm"
  use_lstm: true
  hidden_size: 128
  lstm_hidden_size: 128

# Training configuration
episodes: 10
parallel: 10
algorithm: ppo
device: cpu  # Recommended for LSTM
```

### Production Configuration (`config/train_config_long.yml`)

```yaml
# Sequence learning configuration - Production
sequence_learning:
  enabled: true
  bptt_length: 50     # Truncated BPTT for memory efficiency
  grad_clip_norm: 10.0

# Network configuration
network:
  type: "attention"
  use_lstm: true
  use_attention: true
  hidden_size: 256
  lstm_hidden_size: 256

# Training configuration
episodes: 1000
parallel: 100
algorithm: ppo
device: cpu  # Recommended for LSTM
```

## Device Recommendations

### CPU (Recommended)
```yaml
device: cpu
```
- **Pros**: Full compatibility, stable LSTM gradients
- **Cons**: Slower training
- **Use**: Always recommended for LSTM sequence learning

### CUDA (Untested)
```yaml
device: cuda
```
- **Pros**: Fast training
- **Cons**: Requires NVIDIA GPU, compatibility unknown
- **Use**: Test carefully before production use

### MPS (Not Recommended)
```yaml
device: mps
```
- **Pros**: Fast on Apple Silicon
- **Cons**: Known PyTorch bug with LSTM gradients
- **Status**: Avoid until PyTorch fixes upstream issue

## Algorithm Selection

The training script automatically selects algorithms based on configuration:

### Automatic Selection Logic
```python
if sequence_learning.enabled and network.use_lstm:
    # Uses SequencePPOAlgorithm or SequenceReinforceAlgorithm
    algorithm = SequencePPOAlgorithm(...)
else:
    # Uses standard algorithms
    algorithm = PPOAlgorithm(...)
```

### Manual Override
You can force standard algorithms by setting:
```yaml
sequence_learning:
  enabled: false  # Forces standard algorithms even with LSTM
```

## Common Configuration Patterns

### 1. Quick Testing
```yaml
episodes: 1
parallel: 1
sequence_learning:
  enabled: true
  bptt_length: 0
device: cpu
```

### 2. Development Training
```yaml
episodes: 100
parallel: 10
sequence_learning:
  enabled: true
  bptt_length: 20
  grad_clip_norm: 5.0
device: cpu
```

### 3. Production Training
```yaml
episodes: 1000
parallel: 100
sequence_learning:
  enabled: true
  bptt_length: 50
  grad_clip_norm: 10.0
device: cpu
checkpoint_interval: 100
```

### 4. Memory-Constrained Environment
```yaml
sequence_learning:
  enabled: true
  bptt_length: 10    # Short sequences
  grad_clip_norm: 3.0
parallel: 5          # Fewer parallel environments
```

## Troubleshooting

### Memory Issues
- **Symptom**: Out of memory errors
- **Solution**: Reduce `bptt_length` or `parallel` environments
```yaml
sequence_learning:
  bptt_length: 10  # Reduce from default
parallel: 5        # Reduce parallel envs
```

### Training Instability
- **Symptom**: Loss becomes NaN or explodes
- **Solution**: Reduce gradient clipping norm
```yaml
sequence_learning:
  grad_clip_norm: 2.0  # Reduce from default 5.0
```

### MPS Device Errors
- **Symptom**: `Assertion failed: (shape4.size() >= 3)`
- **Solution**: Switch to CPU device
```yaml
device: cpu  # Instead of mps
```

### Slow Training
- **Symptom**: Very slow episode completion
- **Solution**: Optimize configuration
```yaml
sequence_learning:
  bptt_length: 20  # Use truncated BPTT
parallel: 20       # Increase parallel envs (if memory allows)
```

## Best Practices

### 1. Start Simple
- Begin with default values
- Test with small episode counts
- Gradually increase complexity

### 2. Monitor Memory Usage
- Watch for OOM errors
- Adjust `bptt_length` based on available memory
- Use fewer parallel environments if needed

### 3. Gradient Monitoring
- Start with conservative gradient clipping
- Increase gradually if training is too slow
- Decrease if experiencing instability

### 4. Device Selection
- Always use CPU for LSTM training
- Test CUDA carefully if available
- Avoid MPS until PyTorch fixes are available

### 5. Configuration Validation
Validate your configuration before long training runs:
```bash
# Test configuration with minimal training
python train.py --config your_config.yml --episodes 1 --parallel 1
```

## Migration from Standard Algorithms

### Existing Configurations
If you have existing configurations, add the sequence learning section:

```yaml
# Add this to existing config
sequence_learning:
  enabled: true
  bptt_length: 0
  grad_clip_norm: 5.0

# Ensure LSTM is enabled
network:
  use_lstm: true
```

### Compatibility
- Standard algorithms still work unchanged
- Sequence learning only activates with LSTM networks
- All existing command-line options remain valid

## Performance Tuning

### Memory vs Quality Trade-offs
- **Full Episode BPTT** (`bptt_length: 0`): Best quality, high memory
- **Truncated BPTT** (`bptt_length: 20-50`): Good quality, moderate memory  
- **Short BPTT** (`bptt_length: 10`): Lower quality, low memory

### Training Speed Optimization
1. **Parallel Environments**: Increase if memory allows
2. **BPTT Length**: Shorter sequences train faster
3. **Device**: CUDA > CPU > MPS (when working)
4. **Batch Size**: Larger batches more efficient

## Example Commands

```bash
# Basic sequence learning
python train.py --config config/train_config.yml

# Production training with CPU
python train.py --config config/train_config_long.yml --device cpu

# Override BPTT length
python train.py --config config/train_config.yml --episodes 100

# Quick testing
python train.py --episodes 1 --parallel 1 --device cpu
```