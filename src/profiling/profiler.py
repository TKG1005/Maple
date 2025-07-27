"""Core performance profiler implementation."""

from __future__ import annotations

import time
import threading
import psutil
import torch
from contextlib import contextmanager
from typing import Optional, Dict, Any
from .metrics import ProfileMetrics, SystemInfo


class PerformanceProfiler:
    """Thread-safe performance profiler for training bottleneck analysis."""
    
    def __init__(self, enabled: bool = True, device: Optional[torch.device] = None):
        self.enabled = enabled
        self.device = device or torch.device('cpu')
        self.metrics = ProfileMetrics()
        self.system_info = SystemInfo.collect(self.device) if enabled else None
        self._lock = threading.Lock()
        self._episode_start_time: Optional[float] = None
        self._memory_monitor = MemoryMonitor() if enabled else None
        
    def start_episode(self) -> None:
        """Mark the start of an episode."""
        if not self.enabled:
            return
        
        with self._lock:
            self._episode_start_time = time.perf_counter()
            if self._memory_monitor:
                self._memory_monitor.reset()
    
    def end_episode(self) -> None:
        """Mark the end of an episode and update metrics."""
        if not self.enabled or self._episode_start_time is None:
            return
        
        with self._lock:
            episode_duration = time.perf_counter() - self._episode_start_time
            self.metrics.episode_total += episode_duration
            self.metrics.episode_count += 1
            
            # Update system metrics
            if self._memory_monitor:
                self.metrics.memory_usage_peak = max(
                    self.metrics.memory_usage_peak,
                    self._memory_monitor.get_peak_usage()
                )
                self.metrics.cpu_usage_avg = self._update_average(
                    self.metrics.cpu_usage_avg,
                    self._memory_monitor.get_cpu_usage(),
                    self.metrics.episode_count
                )
                
                if self.device.type == 'cuda':
                    self.metrics.gpu_usage_avg = self._update_average(
                        self.metrics.gpu_usage_avg,
                        self._memory_monitor.get_gpu_usage(),
                        self.metrics.episode_count
                    )
            
            self._episode_start_time = None
    
    @contextmanager
    def profile(self, category: str):
        """Context manager for profiling a specific operation."""
        if not self.enabled:
            yield
            return
        
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            with self._lock:
                self.metrics.add_timing(category, duration)
    
    def record_timing(self, category: str, duration: float) -> None:
        """Record a timing measurement for a category."""
        if not self.enabled:
            return
        
        with self._lock:
            self.metrics.add_timing(category, duration)
    
    def get_metrics(self) -> ProfileMetrics:
        """Get a copy of current metrics."""
        with self._lock:
            # Create a deep copy of the metrics
            new_metrics = ProfileMetrics()
            new_metrics.merge(self.metrics)
            return new_metrics
    
    def get_system_info(self) -> Optional[SystemInfo]:
        """Get system information."""
        return self.system_info
    
    def reset(self) -> None:
        """Reset all metrics."""
        if not self.enabled:
            return
        
        with self._lock:
            self.metrics = ProfileMetrics()
            if self._memory_monitor:
                self._memory_monitor.reset()
    
    def merge(self, other: PerformanceProfiler) -> None:
        """Merge metrics from another profiler instance."""
        if not self.enabled or not other.enabled:
            return
        
        with self._lock:
            self.metrics.merge(other.metrics)
    
    @staticmethod
    def _update_average(current_avg: float, new_value: float, count: int) -> float:
        """Update running average with new value."""
        if count <= 1:
            return new_value
        return ((current_avg * (count - 1)) + new_value) / count


class ProfilerContext:
    """Context manager for easy profiling of code blocks."""
    
    def __init__(self, profiler: PerformanceProfiler, category: str):
        self.profiler = profiler
        self.category = category
        self.start_time = None
    
    def __enter__(self):
        if self.profiler.enabled:
            self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler.enabled and self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.profiler.record_timing(self.category, duration)


class MemoryMonitor:
    """Monitor system memory and GPU usage."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset monitoring statistics."""
        self._peak_memory = 0.0
        self._cpu_usage = 0.0
        self._gpu_usage = 0.0
        self._sample_count = 0
    
    def sample(self) -> None:
        """Take a sample of current system usage."""
        # Memory usage
        memory = psutil.virtual_memory()
        current_memory = (memory.total - memory.available) / (1024**3)  # GB
        self._peak_memory = max(self._peak_memory, current_memory)
        
        # CPU usage (instantaneous)
        self._cpu_usage = ((self._cpu_usage * self._sample_count) + psutil.cpu_percent()) / (self._sample_count + 1)
        
        # GPU usage (if CUDA available)
        if torch.cuda.is_available():
            try:
                gpu_usage = torch.cuda.utilization()
                self._gpu_usage = ((self._gpu_usage * self._sample_count) + gpu_usage) / (self._sample_count + 1)
            except Exception:
                pass  # GPU monitoring not available
        
        self._sample_count += 1
    
    def get_peak_usage(self) -> float:
        """Get peak memory usage in GB."""
        self.sample()  # Take final sample
        return self._peak_memory
    
    def get_cpu_usage(self) -> float:
        """Get average CPU usage percentage."""
        self.sample()  # Take final sample
        return self._cpu_usage
    
    def get_gpu_usage(self) -> float:
        """Get average GPU usage percentage."""
        self.sample()  # Take final sample
        return self._gpu_usage


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_global_profiler() -> Optional[PerformanceProfiler]:
    """Get the global profiler instance."""
    return _global_profiler


def set_global_profiler(profiler: PerformanceProfiler) -> None:
    """Set the global profiler instance."""
    global _global_profiler
    _global_profiler = profiler


def profile(category: str):
    """Decorator for profiling functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_global_profiler()
            if profiler and profiler.enabled:
                with profiler.profile(category):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator