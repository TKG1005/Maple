"""Tests for optimizer utilities including device transfer functionality."""

import tempfile
import torch
import pytest
from pathlib import Path

from src.utils.optimizer_utils import (
    transfer_optimizer_state_to_device,
    save_training_state,
    load_training_state,
    create_scheduler
)


class SimpleNet(torch.nn.Module):
    """Simple network for testing."""
    
    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def sample_networks():
    """Create sample networks for testing."""
    policy_net = SimpleNet(10, 5)
    value_net = SimpleNet(10, 1)
    return policy_net, value_net


@pytest.fixture
def sample_optimizer_and_scheduler(sample_networks):
    """Create sample optimizer and scheduler."""
    policy_net, value_net = sample_networks
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    return optimizer, scheduler


class TestOptimizerStateTransfer:
    """Test optimizer state transfer functionality."""
    
    def test_transfer_optimizer_to_cpu(self, sample_optimizer_and_scheduler):
        """Test transferring optimizer state to CPU."""
        optimizer, _ = sample_optimizer_and_scheduler
        
        # Create some optimizer state by doing a forward/backward pass
        policy_net, value_net = _sample_networks()
        x = torch.randn(4, 10)
        loss = policy_net(x).sum() + value_net(x).sum()
        loss.backward()
        optimizer.step()
        
        # Transfer to CPU (should work regardless of current device)
        transfer_optimizer_state_to_device(optimizer, torch.device("cpu"))
        
        # Check that all state tensors are on CPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    assert v.device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_transfer_optimizer_to_cuda(self, sample_optimizer_and_scheduler):
        """Test transferring optimizer state to CUDA."""
        optimizer, _ = sample_optimizer_and_scheduler
        
        # Move networks to CUDA
        policy_net, value_net = _sample_networks()
        policy_net = policy_net.cuda()
        value_net = value_net.cuda()
        
        # Create some optimizer state
        x = torch.randn(4, 10).cuda()
        loss = policy_net(x).sum() + value_net(x).sum()
        loss.backward()
        optimizer.step()
        
        # Transfer to CUDA
        transfer_optimizer_state_to_device(optimizer, torch.device("cuda"))
        
        # Check that all state tensors are on CUDA
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    assert v.device.type == "cuda"


class TestSaveLoadTrainingState:
    """Test saving and loading complete training state."""
    
    def test_save_and_load_basic_state(self, sample_networks, sample_optimizer_and_scheduler):
        """Test basic save and load functionality."""
        policy_net, value_net = sample_networks
        optimizer, scheduler = sample_optimizer_and_scheduler
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name
        
        try:
            # Save state
            save_training_state(
                filepath,
                policy_net,
                value_net,
                optimizer,
                scheduler,
                episode=42
            )
            
            # Create new networks and optimizer
            new_policy_net = SimpleNet(10, 5)
            new_value_net = SimpleNet(10, 1)
            new_params = list(new_policy_net.parameters()) + list(new_value_net.parameters())
            new_optimizer = torch.optim.Adam(new_params, lr=0.001)
            new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10, gamma=0.9)
            
            # Load state
            state_info = load_training_state(
                filepath,
                new_policy_net,
                new_value_net,
                new_optimizer,
                new_scheduler,
                device=torch.device("cpu")
            )
            
            # Check that state was loaded correctly
            assert state_info["episode"] == 42
            assert state_info["has_optimizer"] is True
            assert state_info["has_scheduler"] is True
            
            # Check that network weights match
            for p1, p2 in zip(policy_net.parameters(), new_policy_net.parameters()):
                assert torch.allclose(p1, p2)
            
            for p1, p2 in zip(value_net.parameters(), new_value_net.parameters()):
                assert torch.allclose(p1, p2)
                
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_save_without_scheduler(self, sample_networks, sample_optimizer_and_scheduler):
        """Test saving state without scheduler."""
        policy_net, value_net = sample_networks
        optimizer, _ = sample_optimizer_and_scheduler
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name
        
        try:
            # Save state without scheduler
            save_training_state(
                filepath,
                policy_net,
                value_net,
                optimizer,
                scheduler=None,
                episode=10
            )
            
            # Create new networks and optimizer
            new_policy_net = SimpleNet(10, 5)
            new_value_net = SimpleNet(10, 1)
            new_params = list(new_policy_net.parameters()) + list(new_value_net.parameters())
            new_optimizer = torch.optim.Adam(new_params, lr=0.001)
            
            # Load state
            state_info = load_training_state(
                filepath,
                new_policy_net,
                new_value_net,
                new_optimizer,
                scheduler=None,
                device=torch.device("cpu")
            )
            
            # Check state
            assert state_info["episode"] == 10
            assert state_info["has_optimizer"] is True
            assert state_info["has_scheduler"] is False
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_load_legacy_format(self, sample_networks):
        """Test loading legacy checkpoint format."""
        policy_net, value_net = sample_networks
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name
        
        try:
            # Save legacy format
            legacy_state = {
                "policy": policy_net.state_dict(),
                "value": value_net.state_dict(),
            }
            torch.save(legacy_state, filepath)
            
            # Create new networks and optimizer
            new_policy_net = SimpleNet(10, 5)
            new_value_net = SimpleNet(10, 1)
            new_params = list(new_policy_net.parameters()) + list(new_value_net.parameters())
            new_optimizer = torch.optim.Adam(new_params, lr=0.001)
            
            # Load state (should handle legacy format gracefully)
            state_info = load_training_state(
                filepath,
                new_policy_net,
                new_value_net,
                new_optimizer,
                scheduler=None,
                device=torch.device("cpu")
            )
            
            # Check that networks loaded but no optimizer/scheduler
            assert state_info["episode"] == 0  # Default for legacy
            assert state_info["has_optimizer"] is False
            assert state_info["has_scheduler"] is False
            
            # Network weights should match
            for p1, p2 in zip(policy_net.parameters(), new_policy_net.parameters()):
                assert torch.allclose(p1, p2)
                
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_transfer_during_load(self, sample_networks):
        """Test device transfer when loading checkpoint."""
        policy_net, value_net = sample_networks
        
        # Move networks to CUDA
        policy_net = policy_net.cuda()
        value_net = value_net.cuda()
        
        # Create optimizer with CUDA tensors
        params = list(policy_net.parameters()) + list(value_net.parameters())
        optimizer = torch.optim.Adam(params, lr=0.001)
        
        # Create some state by doing an optimization step
        x = torch.randn(4, 10).cuda()
        loss = policy_net(x).sum() + value_net(x).sum()
        loss.backward()
        optimizer.step()
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            filepath = f.name
        
        try:
            # Save state on CUDA
            save_training_state(
                filepath,
                policy_net,
                value_net,
                optimizer,
                scheduler=None,
                episode=5
            )
            
            # Load on CPU
            new_policy_net = SimpleNet(10, 5)  # On CPU
            new_value_net = SimpleNet(10, 1)   # On CPU
            new_params = list(new_policy_net.parameters()) + list(new_value_net.parameters())
            new_optimizer = torch.optim.Adam(new_params, lr=0.001)
            
            # Load state and transfer to CPU
            state_info = load_training_state(
                filepath,
                new_policy_net,
                new_value_net,
                new_optimizer,
                scheduler=None,
                device=torch.device("cpu")
            )
            
            # Check that everything is on CPU now
            for param in new_policy_net.parameters():
                assert param.device.type == "cpu"
            for param in new_value_net.parameters():
                assert param.device.type == "cpu"
                
            # Check optimizer state is on CPU
            for state in new_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        assert v.device.type == "cpu"
                        
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestSchedulerCreation:
    """Test scheduler creation functionality."""
    
    def test_create_no_scheduler(self, sample_optimizer_and_scheduler):
        """Test creating no scheduler."""
        optimizer, _ = sample_optimizer_and_scheduler
        
        scheduler = create_scheduler(optimizer, "none")
        assert scheduler is None
        
        scheduler = create_scheduler(optimizer, None)
        assert scheduler is None
    
    def test_create_step_scheduler(self, sample_optimizer_and_scheduler):
        """Test creating step scheduler."""
        optimizer, _ = sample_optimizer_and_scheduler
        
        scheduler = create_scheduler(optimizer, "step", step_size=100, gamma=0.5)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 100
        assert scheduler.gamma == 0.5
    
    def test_create_exponential_scheduler(self, sample_optimizer_and_scheduler):
        """Test creating exponential scheduler."""
        optimizer, _ = sample_optimizer_and_scheduler
        
        scheduler = create_scheduler(optimizer, "exponential", gamma=0.95)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        assert scheduler.gamma == 0.95
    
    def test_create_cosine_scheduler(self, sample_optimizer_and_scheduler):
        """Test creating cosine scheduler."""
        optimizer, _ = sample_optimizer_and_scheduler
        
        scheduler = create_scheduler(optimizer, "cosine", T_max=1000, eta_min=1e-6)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 1000
        assert scheduler.eta_min == 1e-6
    
    def test_create_reduce_on_plateau_scheduler(self, sample_optimizer_and_scheduler):
        """Test creating ReduceLROnPlateau scheduler."""
        optimizer, _ = sample_optimizer_and_scheduler
        
        scheduler = create_scheduler(
            optimizer, 
            "reduce_on_plateau", 
            mode="min", 
            factor=0.2, 
            patience=5
        )
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.mode == "min"
        assert scheduler.factor == 0.2
        assert scheduler.patience == 5
    
    def test_unknown_scheduler_type(self, sample_optimizer_and_scheduler):
        """Test error handling for unknown scheduler type."""
        optimizer, _ = sample_optimizer_and_scheduler
        
        with pytest.raises(ValueError, match="Unknown scheduler type"):
            create_scheduler(optimizer, "unknown_scheduler")


def _sample_networks():
    """Helper function for fixtures that use it as a function."""
    policy_net = SimpleNet(10, 5)
    value_net = SimpleNet(10, 1)
    return policy_net, value_net