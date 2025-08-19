"""Performance metrics collection and system information."""

from __future__ import annotations

import platform
import psutil
import time
import torch
from dataclasses import dataclass, field
import logging
import random
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class SystemInfo:
    """System hardware and software information."""
    platform: str
    platform_version: str
    architecture: str
    processor: str
    cpu_count: int
    cpu_freq_max: float
    memory_total: float  # GB
    memory_available: float  # GB
    python_version: str
    torch_version: str
    device_type: str
    device_name: str
    cuda_version: Optional[str] = None
    mps_available: bool = False
    
    @classmethod
    def collect(cls, device: torch.device) -> SystemInfo:
        """Collect current system information."""
        # CPU info
        cpu_freq = psutil.cpu_freq()
        cpu_freq_max = cpu_freq.max if cpu_freq else 0.0
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_total = memory.total / (1024**3)  # Convert to GB
        memory_available = memory.available / (1024**3)
        
        # Device info
        device_type = str(device.type)
        if device.type == 'cuda':
            device_name = torch.cuda.get_device_name(device)
            cuda_version = torch.version.cuda
        elif device.type == 'mps':
            device_name = "Apple Metal GPU"
            cuda_version = None
        else:
            device_name = "CPU"
            cuda_version = None
        
        # Platform-specific processor info
        try:
            if platform.system() == "Darwin":
                # macOS: Try to get Apple Silicon info
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                processor = result.stdout.strip() if result.returncode == 0 else platform.processor()
            else:
                processor = platform.processor()
        except Exception:
            processor = platform.processor()
        
        return cls(
            platform=platform.system(),
            platform_version=platform.version(),
            architecture=platform.machine(),
            processor=processor,
            cpu_count=psutil.cpu_count(),
            cpu_freq_max=cpu_freq_max,
            memory_total=memory_total,
            memory_available=memory_available,
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            device_type=device_type,
            device_name=device_name,
            cuda_version=cuda_version,
            mps_available=torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'platform': self.platform,
            'platform_version': self.platform_version,
            'architecture': self.architecture,
            'processor': self.processor,
            'cpu_count': self.cpu_count,
            'cpu_freq_max': self.cpu_freq_max,
            'memory_total': self.memory_total,
            'memory_available': self.memory_available,
            'python_version': self.python_version,
            'torch_version': self.torch_version,
            'device_type': self.device_type,
            'device_name': self.device_name,
            'cuda_version': self.cuda_version,
            'mps_available': self.mps_available
        }


@dataclass
class ProfileMetrics:
    """Performance metrics for a single episode or training session."""
    
    # Episode timing
    episode_total: float = 0.0
    episode_count: int = 0
    
    # Environment operations
    env_reset: float = 0.0
    env_step: float = 0.0
    env_close: float = 0.0
    
    # Battle operations  
    battle_init: float = 0.0
    battle_progress: float = 0.0
    battle_websocket: float = 0.0
    
    # Agent operations
    agent_action_selection: float = 0.0
    agent_value_calculation: float = 0.0
    
    # Learning operations
    gradient_calculation: float = 0.0
    optimizer_step: float = 0.0
    loss_calculation: float = 0.0
    
    # I/O operations
    model_save: float = 0.0
    model_load: float = 0.0
    data_logging: float = 0.0
    
    # Memory operations
    tensor_operations: float = 0.0
    device_transfer: float = 0.0
    
    # Other operations
    reward_calculation: float = 0.0
    state_observation: float = 0.0
    action_masking: float = 0.0
    
    # System metrics
    memory_usage_peak: float = 0.0  # GB
    cpu_usage_avg: float = 0.0  # %
    gpu_usage_avg: float = 0.0  # %
    
    # Detailed breakdowns
    timings: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_timing(self, category: str, duration: float) -> None:
        """Add a timing measurement to a category."""
        if category not in self.timings:
            self.timings[category] = []
        self.timings[category].append(duration)
        
        # Update corresponding aggregate field
        if hasattr(self, category):
            setattr(self, category, getattr(self, category) + duration)
    
    def get_average_timing(self, category: str) -> float:
        """Get average timing for a category."""
        if category not in self.timings or not self.timings[category]:
            return 0.0
        return sum(self.timings[category]) / len(self.timings[category])
    
    def get_total_timing(self, category: str) -> float:
        """Get total timing for a category."""
        if category not in self.timings:
            return 0.0
        return sum(self.timings[category])
    
    def merge(self, other: ProfileMetrics) -> None:
        """Merge another ProfileMetrics instance into this one."""
        # Merge episode counts
        self.episode_count += other.episode_count
        
        # Merge timing totals
        for attr in ['episode_total', 'env_reset', 'env_step', 'env_close',
                     'battle_init', 'battle_progress', 'battle_websocket',
                     'agent_action_selection', 'agent_value_calculation',
                     'gradient_calculation', 'optimizer_step', 'loss_calculation',
                     'model_save', 'model_load', 'data_logging',
                     'tensor_operations', 'device_transfer',
                     'reward_calculation', 'state_observation', 'action_masking']:
            setattr(self, attr, getattr(self, attr) + getattr(other, attr))
        
        # Merge system metrics (take max for memory, average for others)
        self.memory_usage_peak = max(self.memory_usage_peak, other.memory_usage_peak)
        if other.episode_count > 0:
            total_episodes = self.episode_count
            if total_episodes > 0:
                self.cpu_usage_avg = ((self.cpu_usage_avg * (total_episodes - other.episode_count)) + 
                                    (other.cpu_usage_avg * other.episode_count)) / total_episodes
                self.gpu_usage_avg = ((self.gpu_usage_avg * (total_episodes - other.episode_count)) + 
                                    (other.gpu_usage_avg * other.episode_count)) / total_episodes
        
        # Merge detailed timings
        for category, timings in other.timings.items():
            if category not in self.timings:
                self.timings[category] = []
            self.timings[category].extend(timings)
    
    def get_averages_per_episode(self) -> Dict[str, float]:
        """Get average timings per episode."""
        if self.episode_count == 0:
            return {}
        
        return {
            'episode_total': self.episode_total / self.episode_count,
            'env_reset': self.env_reset / self.episode_count,
            'env_step': self.env_step / self.episode_count,
            'env_close': self.env_close / self.episode_count,
            'battle_init': self.battle_init / self.episode_count,
            'battle_progress': self.battle_progress / self.episode_count,
            'battle_websocket': self.battle_websocket / self.episode_count,
            'agent_action_selection': self.agent_action_selection / self.episode_count,
            'agent_value_calculation': self.agent_value_calculation / self.episode_count,
            'gradient_calculation': self.gradient_calculation / self.episode_count,
            'optimizer_step': self.optimizer_step / self.episode_count,
            'loss_calculation': self.loss_calculation / self.episode_count,
            'model_save': self.model_save / self.episode_count,
            'model_load': self.model_load / self.episode_count,
            'data_logging': self.data_logging / self.episode_count,
            'tensor_operations': self.tensor_operations / self.episode_count,
            'device_transfer': self.device_transfer / self.episode_count,
            'reward_calculation': self.reward_calculation / self.episode_count,
            'state_observation': self.state_observation / self.episode_count,
            'action_masking': self.action_masking / self.episode_count,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'episode_count': self.episode_count,
            'totals': {
                'episode_total': self.episode_total,
                'env_reset': self.env_reset,
                'env_step': self.env_step,
                'env_close': self.env_close,
                'battle_init': self.battle_init,
                'battle_progress': self.battle_progress,
                'battle_websocket': self.battle_websocket,
                'agent_action_selection': self.agent_action_selection,
                'agent_value_calculation': self.agent_value_calculation,
                'gradient_calculation': self.gradient_calculation,
                'optimizer_step': self.optimizer_step,
                'loss_calculation': self.loss_calculation,
                'model_save': self.model_save,
                'model_load': self.model_load,
                'data_logging': self.data_logging,
                'tensor_operations': self.tensor_operations,
                'device_transfer': self.device_transfer,
                'reward_calculation': self.reward_calculation,
                'state_observation': self.state_observation,
                'action_masking': self.action_masking,
            },
            'averages_per_episode': self.get_averages_per_episode(),
            'system_metrics': {
                'memory_usage_peak': self.memory_usage_peak,
                'cpu_usage_avg': self.cpu_usage_avg,
                'gpu_usage_avg': self.gpu_usage_avg,
            },
            'detailed_timings': {
                category: {
                    'count': len(timings),
                    'total': sum(timings),
                    'average': sum(timings) / len(timings) if timings else 0.0,
                    'min': min(timings) if timings else 0.0,
                    'max': max(timings) if timings else 0.0,
                } for category, timings in self.timings.items()
            }
        }
        return result


# ---- Lightweight metric emitter (Step 6) ----
_logger = logging.getLogger(__name__)


def _sanitize_key(k: str) -> str:
    try:
        k = str(k)
        return "".join(ch if (ch.isalnum() or ch in "_-:") else "_" for ch in k)
    except Exception:
        return "key"


def _coerce_value(v: Any) -> str:
    try:
        if v is None:
            return "none"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return ("%.3f" % v).rstrip("0").rstrip(".")
        s = str(v)
        if not s:
            return ""  # allow empty
        s = s.replace("\n", " ").replace("\t", " ")
        if len(s) > 120:
            s = s[:117] + "..."
        return s
    except Exception:
        return "error"


def _format_metric_line(tag: str, fields: Dict[str, Any]) -> str:
    parts = [f"[METRIC] tag={_coerce_value(tag)}"]
    for k in sorted(fields.keys()):
        parts.append(f"{_sanitize_key(k)}={_coerce_value(fields[k])}")
    return " ".join(parts)


def emit_metric(tag: str, /, *, sample: float = 1.0, **fields: Any) -> None:
    try:
        if sample < 1.0:
            if sample <= 0.0:
                return
            if random.random() > sample:
                return
        if not _logger.isEnabledFor(logging.INFO):
            return
        line = _format_metric_line(tag, fields)
        _logger.info(line)
    except Exception:
        # Swallow all exceptions to avoid impacting control flow
        pass
