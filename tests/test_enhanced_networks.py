import pytest
import torch
import gymnasium as gym
from src.agents.policy_network import PolicyNetwork
from src.agents.value_network import ValueNetwork
from src.agents.enhanced_networks import (
    LSTMPolicyNetwork, LSTMValueNetwork,
    AttentionPolicyNetwork, AttentionValueNetwork,
    MultiHeadAttention
)


class TestEnhancedNetworks:
    """Test suite for enhanced neural networks."""
    
    @pytest.fixture
    def observation_space(self):
        return gym.spaces.Box(low=-1, high=1, shape=(64,))
    
    @pytest.fixture
    def action_space(self):
        return gym.spaces.Discrete(10)
    
    def test_original_networks_2layer(self, observation_space, action_space):
        """Test that original networks work with 2-layer option."""
        # Test PolicyNetwork with 2-layer
        policy_net = PolicyNetwork(observation_space, action_space, use_2layer=True)
        obs = torch.randn(32, 64)
        output = policy_net(obs)
        assert output.shape == (32, 10)
        
        # Test ValueNetwork with 2-layer
        value_net = ValueNetwork(observation_space, use_2layer=True)
        output = value_net(obs)
        assert output.shape == (32,)
        
        # Test backward compatibility with 1-layer
        policy_net_1layer = PolicyNetwork(observation_space, action_space, use_2layer=False)
        value_net_1layer = ValueNetwork(observation_space, use_2layer=False)
        
        output_1layer = policy_net_1layer(obs)
        assert output_1layer.shape == (32, 10)
        
        output_1layer = value_net_1layer(obs)
        assert output_1layer.shape == (32,)
    
    def test_lstm_networks(self, observation_space, action_space):
        """Test LSTM networks."""
        # Test LSTMPolicyNetwork
        lstm_policy = LSTMPolicyNetwork(observation_space, action_space, use_lstm=True)
        obs = torch.randn(32, 64)
        output = lstm_policy(obs)
        assert output.shape == (32, 10)
        
        # Test with sequence input
        seq_obs = torch.randn(32, 5, 64)  # batch, seq_len, obs_dim
        output = lstm_policy(seq_obs)
        assert output.shape == (32, 10)
        
        # Test LSTMValueNetwork
        lstm_value = LSTMValueNetwork(observation_space, use_lstm=True)
        output = lstm_value(obs)
        assert output.shape == (32,)
        
        # Test hidden state initialization
        hidden = lstm_policy.init_hidden(32, torch.device('cpu'))
        assert len(hidden) == 2
        assert hidden[0].shape == (1, 32, 128)
        assert hidden[1].shape == (1, 32, 128)
        
        # Test hidden state reset
        lstm_policy.reset_hidden()
        assert lstm_policy.hidden_state is None
    
    def test_attention_mechanism(self):
        """Test MultiHeadAttention layer."""
        attention = MultiHeadAttention(embed_dim=128, num_heads=4)
        x = torch.randn(32, 10, 128)  # batch, seq_len, embed_dim
        
        output = attention(x)
        assert output.shape == (32, 10, 128)
        
        # Test with mask
        mask = torch.ones(32, 4, 10, 10)  # batch, heads, seq_len, seq_len
        output = attention(x, mask)
        assert output.shape == (32, 10, 128)
    
    def test_attention_networks(self, observation_space, action_space):
        """Test attention-based networks."""
        # Test AttentionPolicyNetwork
        attention_policy = AttentionPolicyNetwork(
            observation_space, action_space, use_attention=True
        )
        obs = torch.randn(32, 64)
        output = attention_policy(obs)
        assert output.shape == (32, 10)
        
        # Test AttentionValueNetwork
        attention_value = AttentionValueNetwork(observation_space, use_attention=True)
        output = attention_value(obs)
        assert output.shape == (32,)
        
        # Test combined LSTM + Attention
        combined_policy = AttentionPolicyNetwork(
            observation_space, action_space, 
            use_attention=True, use_lstm=True
        )
        output = combined_policy(obs)
        assert output.shape == (32, 10)
    
    def test_network_configurations(self, observation_space, action_space):
        """Test various network configurations."""
        configs = [
            {"use_2layer": True, "use_lstm": False, "use_attention": False},
            {"use_2layer": False, "use_lstm": True, "use_attention": False},
            {"use_2layer": True, "use_lstm": True, "use_attention": False},
            {"use_2layer": True, "use_lstm": False, "use_attention": True},
            {"use_2layer": True, "use_lstm": True, "use_attention": True},
        ]
        
        for config in configs:
            if config["use_lstm"] or config["use_attention"]:
                # Test enhanced networks
                policy_net = AttentionPolicyNetwork(
                    observation_space, action_space, **config
                )
                value_net = AttentionValueNetwork(observation_space, **config)
            else:
                # Test basic networks
                policy_net = PolicyNetwork(
                    observation_space, action_space, use_2layer=config["use_2layer"]
                )
                value_net = ValueNetwork(
                    observation_space, use_2layer=config["use_2layer"]
                )
            
            obs = torch.randn(16, 64)
            
            policy_output = policy_net(obs)
            assert policy_output.shape == (16, 10)
            
            value_output = value_net(obs)
            assert value_output.shape == (16,)
    
    def test_gradient_flow(self, observation_space, action_space):
        """Test that gradients flow properly through networks."""
        networks = [
            PolicyNetwork(observation_space, action_space, use_2layer=True),
            ValueNetwork(observation_space, use_2layer=True),
            LSTMPolicyNetwork(observation_space, action_space, use_lstm=True),
            LSTMValueNetwork(observation_space, use_lstm=True),
            AttentionPolicyNetwork(observation_space, action_space, use_attention=True),
            AttentionValueNetwork(observation_space, use_attention=True),
        ]
        
        for net in networks:
            obs = torch.randn(8, 64, requires_grad=True)
            output = net(obs)
            
            # Compute a simple loss
            if isinstance(net, (PolicyNetwork, LSTMPolicyNetwork, AttentionPolicyNetwork)):
                loss = output.sum()
            else:
                loss = output.sum()
            
            loss.backward()
            
            # Check that gradients exist
            assert obs.grad is not None
            
            # Check that network parameters have gradients
            for param in net.parameters():
                if param.requires_grad:
                    assert param.grad is not None
    
    def test_device_compatibility(self, observation_space, action_space):
        """Test that networks work on different devices."""
        device = torch.device('cpu')
        
        policy_net = PolicyNetwork(observation_space, action_space, use_2layer=True)
        policy_net.to(device)
        
        obs = torch.randn(4, 64, device=device)
        output = policy_net(obs)
        assert output.device == device
        
        # Test LSTM with device
        lstm_policy = LSTMPolicyNetwork(observation_space, action_space, use_lstm=True)
        lstm_policy.to(device)
        
        hidden = lstm_policy.init_hidden(4, device)
        output = lstm_policy(obs, hidden)
        assert output.device == device
    
    def test_network_parameter_counts(self, observation_space, action_space):
        """Test that 2-layer networks have more parameters than 1-layer."""
        policy_1layer = PolicyNetwork(observation_space, action_space, use_2layer=False)
        policy_2layer = PolicyNetwork(observation_space, action_space, use_2layer=True)
        
        params_1layer = sum(p.numel() for p in policy_1layer.parameters())
        params_2layer = sum(p.numel() for p in policy_2layer.parameters())
        
        assert params_2layer > params_1layer
        
        value_1layer = ValueNetwork(observation_space, use_2layer=False)
        value_2layer = ValueNetwork(observation_space, use_2layer=True)
        
        params_1layer = sum(p.numel() for p in value_1layer.parameters())
        params_2layer = sum(p.numel() for p in value_2layer.parameters())
        
        assert params_2layer > params_1layer
    
    def test_error_handling(self, observation_space, action_space):
        """Test error handling for invalid inputs."""
        with pytest.raises(TypeError):
            PolicyNetwork(gym.spaces.Discrete(10), action_space)
        
        with pytest.raises(TypeError):
            PolicyNetwork(observation_space, gym.spaces.Box(low=0, high=1, shape=(5,)))
        
        with pytest.raises(TypeError):
            ValueNetwork(gym.spaces.Discrete(10))
        
        # Test attention with invalid embed_dim
        with pytest.raises(AssertionError):
            MultiHeadAttention(embed_dim=127, num_heads=4)  # 127 not divisible by 4