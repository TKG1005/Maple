from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np

from .base import BaseAlgorithm


class SequencePPOAlgorithm(BaseAlgorithm):
    """PPO algorithm optimized for LSTM/sequence learning with proper step-by-step processing."""

    def __init__(
        self,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        bptt_length: int = 0,  # 0 means full episode length
        grad_clip_norm: float = 5.0,
    ) -> None:
        self.clip_range = float(clip_range)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.bptt_length = int(bptt_length)
        self.grad_clip_norm = float(grad_clip_norm)

    def _process_sequence_step_by_step(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        obs_sequence: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence step by step to get predictions for each timestep."""
        seq_len = obs_sequence.shape[0]
        
        # Initialize hidden states
        policy_hidden = None
        value_hidden = None
        
        # Collect outputs for each timestep
        all_logits = []
        all_values = []
        
        for t in range(seq_len):
            # Get observation for this timestep
            obs_t = obs_sequence[t:t+1]  # [1, obs_dim]
            
            # Forward pass through networks
            policy_output = policy_net(obs_t, policy_hidden)
            value_output = value_net(obs_t, value_hidden)
            
            # Handle network outputs
            if isinstance(policy_output, tuple):
                logits_t, policy_hidden = policy_output
            else:
                logits_t = policy_output
                
            if isinstance(value_output, tuple):
                value_t, value_hidden = value_output
            else:
                value_t = value_output
            
            # Store outputs
            all_logits.append(logits_t.squeeze(0))  # Remove batch dimension
            # Value network returns scalar, but we need to handle different shapes
            if value_t.dim() == 2:
                # If value_t is [1, 1], squeeze to scalar
                all_values.append(value_t.squeeze())
            elif value_t.dim() == 1:
                # If value_t is [1], squeeze to scalar
                all_values.append(value_t.squeeze(0))
            else:
                # If already scalar
                all_values.append(value_t)
        
        # Stack all outputs
        logits = torch.stack(all_logits, dim=0)  # [seq_len, action_dim]
        values = torch.stack(all_values, dim=0)  # [seq_len]
        
        return logits, values

    def _compute_sequence_loss(
        self,
        policy_net: nn.Module,
        value_net: nn.Module,
        sequence: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss for a single sequence."""
        # Prepare sequence data (convert numpy arrays to tensors)
        obs = torch.as_tensor(sequence["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(sequence["actions"], dtype=torch.int64, device=device)
        old_log_probs = torch.as_tensor(sequence["old_log_probs"], dtype=torch.float32, device=device)
        advantages = torch.as_tensor(sequence["advantages"], dtype=torch.float32, device=device)
        returns = torch.as_tensor(sequence["returns"], dtype=torch.float32, device=device)
        values = torch.as_tensor(sequence.get("values", np.zeros_like(sequence["returns"])), dtype=torch.float32, device=device)
        
        # Process sequence to get predictions for each timestep
        logits, value_estimates = self._process_sequence_step_by_step(
            policy_net, value_net, obs, device
        )
        
        # Compute policy loss
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[torch.arange(len(actions)), actions]
        
        ratio = torch.exp(selected_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Compute value loss
        value_loss = 0.5 * (returns - value_estimates).pow(2).mean()
        
        # Compute entropy loss
        entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Metrics
        metrics = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "total_loss": float(total_loss.detach()),
        }
        
        return total_loss, metrics

    def _split_episode_into_sequences(
        self, 
        batch: Dict[str, torch.Tensor], 
        episode_lengths: List[int]
    ) -> List[Dict[str, torch.Tensor]]:
        """Split episode data into sequences based on BPTT length."""
        sequences = []
        start_idx = 0
        
        for ep_len in episode_lengths:
            episode_data = {
                key: value[start_idx:start_idx + ep_len] 
                for key, value in batch.items()
            }
            
            if self.bptt_length > 0 and ep_len > self.bptt_length:
                # Split long episodes into truncated sequences
                for seq_start in range(0, ep_len, self.bptt_length):
                    seq_end = min(seq_start + self.bptt_length, ep_len)
                    seq_data = {
                        key: value[seq_start:seq_end] 
                        for key, value in episode_data.items()
                        if key != "episode_lengths"  # Skip episode_lengths for subsequences
                    }
                    sequences.append(seq_data)
            else:
                # Use full episode as sequence
                sequences.append(episode_data)
            
            start_idx += ep_len
        
        return sequences

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        """Update model using sequence-based PPO with configurable BPTT length."""
        if optimizer is None:
            return 0.0
            
        # Handle different model formats
        if isinstance(model, (tuple, list)) and len(model) == 2:
            policy_net, value_net = model
        elif isinstance(model, dict):
            policy_net = model["policy"]
            value_net = model["value"]
        else:
            # This is not supported for sequence PPO - we need both networks
            raise ValueError(
                "SequencePPOAlgorithm requires both policy and value networks. "
                "Please pass a tuple (policy_net, value_net) or dict {'policy': policy_net, 'value': value_net}."
            )
        
        device = next(policy_net.parameters()).device
        
        # Extract episode lengths from batch (if available)
        episode_lengths = batch.get("episode_lengths", [len(batch["observations"])])
        if isinstance(episode_lengths, (torch.Tensor, np.ndarray)):
            episode_lengths = episode_lengths.tolist()
        
        # Split batch into sequences
        sequences = self._split_episode_into_sequences(batch, episode_lengths)
        
        total_loss = 0.0
        total_metrics = {}
        
        for seq_idx, sequence in enumerate(sequences):
            # Compute loss for this sequence
            loss, metrics = self._compute_sequence_loss(
                policy_net, value_net, sequence, device
            )
            
            # Accumulate loss
            total_loss += loss
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
        
        # Average loss and metrics
        if len(sequences) > 0:
            total_loss /= len(sequences)
            for key in total_metrics:
                total_metrics[key] /= len(sequences)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.grad_clip_norm > 0:
            if isinstance(model, (tuple, list)):
                # Clip gradients for both networks
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=self.grad_clip_norm)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=self.grad_clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_norm)
        
        optimizer.step()
        
        return float(total_loss.detach())


class SequenceReinforceAlgorithm(BaseAlgorithm):
    """REINFORCE algorithm optimized for LSTM/sequence learning."""

    def __init__(
        self,
        bptt_length: int = 0,  # 0 means full episode length
        grad_clip_norm: float = 5.0,
    ) -> None:
        self.bptt_length = int(bptt_length)
        self.grad_clip_norm = float(grad_clip_norm)

    def _process_sequence_step_by_step(
        self,
        policy_net: nn.Module,
        obs_sequence: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Process a sequence step by step to get predictions for each timestep."""
        seq_len = obs_sequence.shape[0]
        
        # Initialize hidden states
        policy_hidden = None
        
        # Collect outputs for each timestep
        all_logits = []
        
        for t in range(seq_len):
            # Get observation for this timestep
            obs_t = obs_sequence[t:t+1]  # [1, obs_dim]
            
            # Forward pass through network
            policy_output = policy_net(obs_t, policy_hidden)
            
            # Handle network outputs
            if isinstance(policy_output, tuple):
                logits_t, policy_hidden = policy_output
            else:
                logits_t = policy_output
            
            # Store outputs
            all_logits.append(logits_t.squeeze(0))  # Remove batch dimension
        
        # Stack all outputs
        logits = torch.stack(all_logits, dim=0)  # [seq_len, action_dim]
        
        return logits

    def _compute_sequence_loss(
        self,
        policy_net: nn.Module,
        sequence: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute REINFORCE loss for a single sequence."""
        # Prepare sequence data (convert numpy arrays to tensors)
        obs = torch.as_tensor(sequence["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(sequence["actions"], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(sequence["rewards"], dtype=torch.float32, device=device)
        
        # Process sequence to get predictions for each timestep
        logits = self._process_sequence_step_by_step(policy_net, obs, device)
        
        # Compute policy loss
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[torch.arange(len(actions)), actions]
        
        loss = -(selected_log_probs * rewards).mean()
        
        # Metrics
        metrics = {
            "policy_loss": float(loss.detach()),
            "total_loss": float(loss.detach()),
        }
        
        return loss, metrics

    def _split_episode_into_sequences(
        self, 
        batch: Dict[str, torch.Tensor], 
        episode_lengths: List[int]
    ) -> List[Dict[str, torch.Tensor]]:
        """Split episode data into sequences based on BPTT length."""
        sequences = []
        start_idx = 0
        
        for ep_len in episode_lengths:
            episode_data = {
                key: value[start_idx:start_idx + ep_len] 
                for key, value in batch.items()
            }
            
            if self.bptt_length > 0 and ep_len > self.bptt_length:
                # Split long episodes into truncated sequences
                for seq_start in range(0, ep_len, self.bptt_length):
                    seq_end = min(seq_start + self.bptt_length, ep_len)
                    seq_data = {
                        key: value[seq_start:seq_end] 
                        for key, value in episode_data.items()
                        if key != "episode_lengths"  # Skip episode_lengths for subsequences
                    }
                    sequences.append(seq_data)
            else:
                # Use full episode as sequence
                sequences.append(episode_data)
            
            start_idx += ep_len
        
        return sequences

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        """Update model using sequence-based REINFORCE with configurable BPTT length."""
        if optimizer is None:
            return 0.0
        
        # Handle different model formats - for REINFORCE we only need policy network
        if isinstance(model, (tuple, list)):
            policy_net = model[0]  # Use the first network (policy)
        elif isinstance(model, dict):
            policy_net = model["policy"]
        else:
            # Single model case - assume it's the policy network
            policy_net = model
            
        device = next(policy_net.parameters()).device
        
        # Extract episode lengths from batch (if available)
        episode_lengths = batch.get("episode_lengths", [len(batch["observations"])])
        if isinstance(episode_lengths, (torch.Tensor, np.ndarray)):
            episode_lengths = episode_lengths.tolist()
        
        # Split batch into sequences
        sequences = self._split_episode_into_sequences(batch, episode_lengths)
        
        total_loss = 0.0
        total_metrics = {}
        
        for seq_idx, sequence in enumerate(sequences):
            # Compute loss for this sequence
            loss, metrics = self._compute_sequence_loss(policy_net, sequence, device)
            
            # Accumulate loss
            total_loss += loss
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
        
        # Average loss and metrics
        if len(sequences) > 0:
            total_loss /= len(sequences)
            for key in total_metrics:
                total_metrics[key] /= len(sequences)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=self.grad_clip_norm)
        
        optimizer.step()
        
        return float(total_loss.detach())


__all__ = ["SequencePPOAlgorithm", "SequenceReinforceAlgorithm"]