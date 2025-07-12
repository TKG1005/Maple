"""Factory functions for creating neural networks with different architectures."""

from typing import Dict, Any
import gymnasium as gym
from .policy_network import PolicyNetwork
from .value_network import ValueNetwork
from .enhanced_networks import (
    LSTMPolicyNetwork, LSTMValueNetwork,
    AttentionPolicyNetwork, AttentionValueNetwork
)
from .embedding_networks import (
    EmbeddingPolicyNetwork, EmbeddingValueNetwork,
    get_embedding_network_info
)


def create_policy_network(
    observation_space: gym.Space,
    action_space: gym.Space,
    config: Dict[str, Any]
) -> PolicyNetwork:
    """Create a policy network based on configuration.
    
    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        config: Network configuration dictionary
        
    Returns:
        Policy network instance
    """
    network_type = config.get("type", "basic")
    hidden_size = config.get("hidden_size", 128)
    use_2layer = config.get("use_2layer", True)
    
    if network_type == "basic":
        return PolicyNetwork(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            use_2layer=use_2layer
        )
    elif network_type == "lstm":
        use_lstm = config.get("use_lstm", True)
        lstm_hidden_size = config.get("lstm_hidden_size", 128)
        return LSTMPolicyNetwork(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            lstm_hidden_size=lstm_hidden_size,
            use_lstm=use_lstm,
            use_2layer=use_2layer
        )
    elif network_type == "attention":
        use_attention = config.get("use_attention", True)
        use_lstm = config.get("use_lstm", False)
        attention_heads = config.get("attention_heads", 4)
        attention_dropout = config.get("attention_dropout", 0.1)
        return AttentionPolicyNetwork(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            num_heads=attention_heads,
            use_attention=use_attention,
            use_lstm=use_lstm,
            use_2layer=use_2layer
        )
    elif network_type == "embedding":
        embedding_config = config.get("embedding_config", {})
        return EmbeddingPolicyNetwork(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=hidden_size,
            use_2layer=use_2layer,
            embedding_config=embedding_config
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def create_value_network(
    observation_space: gym.Space,
    config: Dict[str, Any]
) -> ValueNetwork:
    """Create a value network based on configuration.
    
    Args:
        observation_space: Environment observation space
        config: Network configuration dictionary
        
    Returns:
        Value network instance
    """
    network_type = config.get("type", "basic")
    hidden_size = config.get("hidden_size", 128)
    use_2layer = config.get("use_2layer", True)
    
    if network_type == "basic":
        return ValueNetwork(
            observation_space=observation_space,
            hidden_size=hidden_size,
            use_2layer=use_2layer
        )
    elif network_type == "lstm":
        use_lstm = config.get("use_lstm", True)
        lstm_hidden_size = config.get("lstm_hidden_size", 128)
        return LSTMValueNetwork(
            observation_space=observation_space,
            hidden_size=hidden_size,
            lstm_hidden_size=lstm_hidden_size,
            use_lstm=use_lstm,
            use_2layer=use_2layer
        )
    elif network_type == "attention":
        use_attention = config.get("use_attention", True)
        use_lstm = config.get("use_lstm", False)
        attention_heads = config.get("attention_heads", 4)
        attention_dropout = config.get("attention_dropout", 0.1)
        return AttentionValueNetwork(
            observation_space=observation_space,
            hidden_size=hidden_size,
            num_heads=attention_heads,
            use_attention=use_attention,
            use_lstm=use_lstm,
            use_2layer=use_2layer
        )
    elif network_type == "embedding":
        embedding_config = config.get("embedding_config", {})
        return EmbeddingValueNetwork(
            observation_space=observation_space,
            hidden_size=hidden_size,
            use_2layer=use_2layer,
            embedding_config=embedding_config
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def get_network_info(network) -> Dict[str, Any]:
    """Get information about a network instance.
    
    Args:
        network: Network instance
        
    Returns:
        Dictionary with network information
    """
    # Check if this is an embedding network
    if isinstance(network, (EmbeddingPolicyNetwork, EmbeddingValueNetwork)):
        return get_embedding_network_info(network)
    
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    info = {
        "type": network.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    
    # Add architecture-specific information
    if hasattr(network, 'use_lstm') and network.use_lstm:
        info["has_lstm"] = True
        info["lstm_hidden_size"] = getattr(network, 'lstm_hidden_size', None)
    
    if hasattr(network, 'use_attention') and network.use_attention:
        info["has_attention"] = True
        attention_layer = getattr(network, 'attention', None)
        if attention_layer is not None:
            info["attention_heads"] = getattr(attention_layer, 'num_heads', None)
    
    return info