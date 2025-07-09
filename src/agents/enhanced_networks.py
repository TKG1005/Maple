import torch
import torch.nn as nn
import gymnasium as gym
from typing import Optional, Tuple


class LSTMPolicyNetwork(nn.Module):
    """Policy network with LSTM for sequential processing."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, 
                 hidden_size: int = 128, lstm_hidden_size: int = 128, 
                 use_lstm: bool = True, use_2layer: bool = True) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be gym.spaces.Box")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("action_space must be gym.spaces.Discrete")
        
        obs_dim = int(observation_space.shape[0])
        action_dim = int(action_space.n)
        
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        
        if use_lstm:
            self.lstm = nn.LSTM(obs_dim, lstm_hidden_size, batch_first=True)
            input_dim = lstm_hidden_size
        else:
            input_dim = obs_dim
        
        if use_2layer:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
            )
        
        self.hidden_state = None
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        if self.use_lstm:
            if x.dim() == 2:
                # Add sequence dimension if not present
                x = x.unsqueeze(1)
            
            if hidden is None:
                hidden = self.init_hidden(x.size(0), x.device)
            
            lstm_out, self.hidden_state = self.lstm(x, hidden)
            # Use the last output of the sequence
            x = lstm_out[:, -1, :]
        
        return self.mlp(x)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def reset_hidden(self):
        """Reset hidden state (call at episode boundaries)."""
        self.hidden_state = None


class LSTMValueNetwork(nn.Module):
    """Value network with LSTM for sequential processing."""
    
    def __init__(self, observation_space: gym.Space, hidden_size: int = 128, 
                 lstm_hidden_size: int = 128, use_lstm: bool = True, use_2layer: bool = True) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be gym.spaces.Box")
        
        obs_dim = int(observation_space.shape[0])
        
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        
        if use_lstm:
            self.lstm = nn.LSTM(obs_dim, lstm_hidden_size, batch_first=True)
            input_dim = lstm_hidden_size
        else:
            input_dim = obs_dim
        
        if use_2layer:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
        
        self.hidden_state = None
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        if self.use_lstm:
            if x.dim() == 2:
                # Add sequence dimension if not present
                x = x.unsqueeze(1)
            
            if hidden is None:
                hidden = self.init_hidden(x.size(0), x.device)
            
            lstm_out, self.hidden_state = self.lstm(x, hidden)
            # Use the last output of the sequence
            x = lstm_out[:, -1, :]
        
        return self.mlp(x).squeeze(-1)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (h0, c0)
    
    def reset_hidden(self):
        """Reset hidden state (call at episode boundaries)."""
        self.hidden_state = None


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out(out)


class AttentionPolicyNetwork(nn.Module):
    """Policy network with attention mechanism."""
    
    def __init__(self, observation_space: gym.Space, action_space: gym.Space,
                 hidden_size: int = 128, num_heads: int = 4, use_attention: bool = True,
                 use_lstm: bool = False, use_2layer: bool = True) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be gym.spaces.Box")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("action_space must be gym.spaces.Discrete")
        
        obs_dim = int(observation_space.shape[0])
        action_dim = int(action_space.n)
        
        self.use_attention = use_attention
        self.use_lstm = use_lstm
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_size)
        
        # LSTM layer (optional)
        if use_lstm:
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.hidden_state = None
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size, num_heads)
            self.norm1 = nn.LayerNorm(hidden_size)
        
        # Output MLP
        if use_2layer:
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
            )
        else:
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_dim),
            )
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM processing
        if self.use_lstm:
            if hidden is None:
                hidden = self.init_hidden(x.size(0), x.device)
            x, self.hidden_state = self.lstm(x, hidden)
        
        # Attention mechanism
        if self.use_attention:
            attended = self.attention(x)
            x = self.norm1(x + attended)  # Residual connection
        
        # Use the last timestep for output
        x = x[:, -1, :]
        
        return self.output_mlp(x)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(1, batch_size, self.input_proj.out_features, device=device)
        c0 = torch.zeros(1, batch_size, self.input_proj.out_features, device=device)
        return (h0, c0)
    
    def reset_hidden(self):
        """Reset hidden state (call at episode boundaries)."""
        if self.use_lstm:
            self.hidden_state = None


class AttentionValueNetwork(nn.Module):
    """Value network with attention mechanism."""
    
    def __init__(self, observation_space: gym.Space, hidden_size: int = 128,
                 num_heads: int = 4, use_attention: bool = True,
                 use_lstm: bool = False, use_2layer: bool = True) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be gym.spaces.Box")
        
        obs_dim = int(observation_space.shape[0])
        
        self.use_attention = use_attention
        self.use_lstm = use_lstm
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_size)
        
        # LSTM layer (optional)
        if use_lstm:
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.hidden_state = None
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = MultiHeadAttention(hidden_size, num_heads)
            self.norm1 = nn.LayerNorm(hidden_size)
        
        # Output MLP
        if use_2layer:
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.output_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM processing
        if self.use_lstm:
            if hidden is None:
                hidden = self.init_hidden(x.size(0), x.device)
            x, self.hidden_state = self.lstm(x, hidden)
        
        # Attention mechanism
        if self.use_attention:
            attended = self.attention(x)
            x = self.norm1(x + attended)  # Residual connection
        
        # Use the last timestep for output
        x = x[:, -1, :]
        
        return self.output_mlp(x).squeeze(-1)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(1, batch_size, self.input_proj.out_features, device=device)
        c0 = torch.zeros(1, batch_size, self.input_proj.out_features, device=device)
        return (h0, c0)
    
    def reset_hidden(self):
        """Reset hidden state (call at episode boundaries)."""
        if self.use_lstm:
            self.hidden_state = None