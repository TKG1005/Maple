"""Test suite for optimizer utilities."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from src.utils.optimizer_utils import (
    save_training_state,
    load_training_state,
    transfer_optimizer_state_to_device,
    create_scheduler,
)


class SimpleNetwork(nn.Module):
    """Simple test network."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 5, output_size: int = 3):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class TestOptimizerUtils(unittest.TestCase):
    """Test optimizer utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.policy_net = SimpleNetwork()
        self.value_net = SimpleNetwork(output_size=1)
        params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        self.optimizer = optim.Adam(params, lr=0.001)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
    
    def test_save_and_load_training_state(self):
        """Test saving and loading complete training state."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Save training state
            episode = 42
            save_training_state(
                checkpoint_path=checkpoint_path,
                episode=episode,
                policy_net=self.policy_net,
                value_net=self.value_net,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            
            # Create new networks and optimizer
            new_policy_net = SimpleNetwork()
            new_value_net = SimpleNetwork(output_size=1)
            new_params = list(new_policy_net.parameters()) + list(new_value_net.parameters())
            new_optimizer = optim.Adam(new_params, lr=0.002)  # Different LR
            new_scheduler = lr_scheduler.StepLR(new_optimizer, step_size=10, gamma=0.5)
            
            # Load training state
            loaded_episode = load_training_state(
                checkpoint_path=checkpoint_path,
                policy_net=new_policy_net,
                value_net=new_value_net,
                optimizer=new_optimizer,
                scheduler=new_scheduler,
                device="cpu",
                reset_optimizer=False,
            )
            
            # Verify episode number
            self.assertEqual(loaded_episode, episode)
            
            # Verify optimizer state was loaded (LR should be original, not new)
            self.assertAlmostEqual(new_optimizer.param_groups[0]['lr'], 0.001, places=5)
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
    
    def test_load_training_state_with_reset_optimizer(self):
        """Test loading training state with optimizer reset."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Save training state
            save_training_state(
                checkpoint_path=checkpoint_path,
                episode=10,
                policy_net=self.policy_net,
                value_net=self.value_net,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            
            # Create new networks and optimizer
            new_policy_net = SimpleNetwork()
            new_value_net = SimpleNetwork(output_size=1)
            new_params = list(new_policy_net.parameters()) + list(new_value_net.parameters())
            new_optimizer = optim.Adam(new_params, lr=0.005)  # Different LR
            new_scheduler = lr_scheduler.StepLR(new_optimizer, step_size=10, gamma=0.5)
            
            # Load training state with optimizer reset
            load_training_state(
                checkpoint_path=checkpoint_path,
                policy_net=new_policy_net,
                value_net=new_value_net,
                optimizer=new_optimizer,
                scheduler=new_scheduler,
                device="cpu",
                reset_optimizer=True,
            )
            
            # Verify optimizer state was NOT loaded (LR should be new, not original)
            self.assertAlmostEqual(new_optimizer.param_groups[0]['lr'], 0.005, places=5)
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
    
    def test_load_legacy_format(self):
        """Test loading legacy checkpoint format."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Save legacy format (single state dict) - use matching network structure
            torch.save(self.policy_net.state_dict(), checkpoint_path)
            
            # Create new networks with same structure
            new_policy_net = SimpleNetwork()  # Same output_size=3
            new_value_net = SimpleNetwork()  # Same output_size=3 for legacy test
            new_params = list(new_policy_net.parameters()) + list(new_value_net.parameters())
            new_optimizer = optim.Adam(new_params, lr=0.001)
            
            # Load should work without errors
            episode = load_training_state(
                checkpoint_path=checkpoint_path,
                policy_net=new_policy_net,
                value_net=new_value_net,
                optimizer=new_optimizer,
                device="cpu",
                reset_optimizer=False,
            )
            
            # Episode should default to 0 for legacy format
            self.assertEqual(episode, 0)
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)
    
    def test_transfer_optimizer_state_to_device(self):
        """Test transferring optimizer state to different device."""
        # Create some optimizer state by running a step
        x = torch.randn(2, 10)
        y = self.policy_net(x)
        loss = y.sum()
        loss.backward()
        self.optimizer.step()
        
        # Verify optimizer has state
        self.assertTrue(len(self.optimizer.state) > 0)
        
        # Test transferring to CPU (should work without errors)
        transfer_optimizer_state_to_device(self.optimizer, "cpu")
        
        # Verify state is still present
        self.assertTrue(len(self.optimizer.state) > 0)
    
    def test_create_scheduler_step(self):
        """Test creating StepLR scheduler."""
        config = {
            "enabled": True,
            "type": "step",
            "step_size": 5,
            "gamma": 0.8,
        }
        
        scheduler = create_scheduler(self.optimizer, config)
        
        self.assertIsInstance(scheduler, lr_scheduler.StepLR)
        self.assertEqual(scheduler.step_size, 5)
        self.assertAlmostEqual(scheduler.gamma, 0.8)
    
    def test_create_scheduler_exponential(self):
        """Test creating ExponentialLR scheduler."""
        config = {
            "enabled": True,
            "type": "exponential",
            "gamma": 0.95,
        }
        
        scheduler = create_scheduler(self.optimizer, config)
        
        self.assertIsInstance(scheduler, lr_scheduler.ExponentialLR)
        self.assertAlmostEqual(scheduler.gamma, 0.95)
    
    def test_create_scheduler_cosine(self):
        """Test creating CosineAnnealingLR scheduler."""
        config = {
            "enabled": True,
            "type": "cosine",
            "T_max": 100,
            "eta_min": 1e-6,
        }
        
        scheduler = create_scheduler(self.optimizer, config)
        
        self.assertIsInstance(scheduler, lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.T_max, 100)
        self.assertAlmostEqual(scheduler.eta_min, 1e-6)
    
    def test_create_scheduler_reduce_on_plateau(self):
        """Test creating ReduceLROnPlateau scheduler."""
        config = {
            "enabled": True,
            "type": "reduce_on_plateau",
            "mode": "max",
            "factor": 0.3,
            "patience": 15,
        }
        
        scheduler = create_scheduler(self.optimizer, config)
        
        self.assertIsInstance(scheduler, lr_scheduler.ReduceLROnPlateau)
        self.assertEqual(scheduler.mode, "max")
        self.assertAlmostEqual(scheduler.factor, 0.3)
        self.assertEqual(scheduler.patience, 15)
    
    def test_create_scheduler_disabled(self):
        """Test creating scheduler when disabled."""
        config = {
            "enabled": False,
            "type": "step",
        }
        
        scheduler = create_scheduler(self.optimizer, config)
        
        self.assertIsNone(scheduler)
    
    def test_create_scheduler_invalid_type(self):
        """Test creating scheduler with invalid type."""
        config = {
            "enabled": True,
            "type": "invalid_type",
        }
        
        with self.assertRaises(ValueError):
            create_scheduler(self.optimizer, config)
    
    def test_save_without_scheduler(self):
        """Test saving training state without scheduler."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            checkpoint_path = f.name
        
        try:
            save_training_state(
                checkpoint_path=checkpoint_path,
                episode=5,
                policy_net=self.policy_net,
                value_net=self.value_net,
                optimizer=self.optimizer,
                scheduler=None,
            )
            
            # Load without scheduler
            new_policy_net = SimpleNetwork()
            new_value_net = SimpleNetwork(output_size=1)
            new_params = list(new_policy_net.parameters()) + list(new_value_net.parameters())
            new_optimizer = optim.Adam(new_params, lr=0.001)
            
            episode = load_training_state(
                checkpoint_path=checkpoint_path,
                policy_net=new_policy_net,
                value_net=new_value_net,
                optimizer=new_optimizer,
                scheduler=None,
                device="cpu",
                reset_optimizer=False,
            )
            
            self.assertEqual(episode, 5)
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()