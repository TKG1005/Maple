"""Performance logging and report generation."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import csv

from .metrics import ProfileMetrics, SystemInfo


class PerformanceLogger:
    """Logger for performance profiling results."""
    
    def __init__(self, log_dir: Path, enabled: bool = True):
        self.log_dir = log_dir
        self.enabled = enabled
        
        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.log_dir / "raw").mkdir(exist_ok=True)
            (self.log_dir / "reports").mkdir(exist_ok=True)
            (self.log_dir / "comparison").mkdir(exist_ok=True)
    
    def log_session(self, 
                   metrics: ProfileMetrics, 
                   system_info: SystemInfo,
                   session_name: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Log a complete profiling session."""
        if not self.enabled:
            return Path()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if session_name:
            filename = f"{session_name}_{timestamp}.json"
        else:
            filename = f"profile_{timestamp}.json"
        
        log_file = self.log_dir / "raw" / filename
        
        data = {
            'timestamp': timestamp,
            'session_name': session_name,
            'metadata': metadata or {},
            'system_info': system_info.to_dict(),
            'metrics': metrics.to_dict()
        }
        
        with open(log_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also create a summary report
        self._create_summary_report(data, log_file.stem)
        
        return log_file
    
    def _create_summary_report(self, data: Dict[str, Any], session_id: str) -> None:
        """Create a human-readable summary report."""
        report_file = self.log_dir / "reports" / f"{session_id}_summary.txt"
        
        system_info = data['system_info']
        metrics = data['metrics']
        
        with open(report_file, 'w') as f:
            f.write("PERFORMANCE PROFILING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Session info
            f.write(f"Session: {data.get('session_name', 'Unknown')}\n")
            f.write(f"Timestamp: {data['timestamp']}\n")
            f.write(f"Episodes: {metrics['episode_count']}\n\n")
            
            # System info
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Platform: {system_info['platform']} {system_info['platform_version']}\n")
            f.write(f"Architecture: {system_info['architecture']}\n")
            f.write(f"Processor: {system_info['processor']}\n")
            f.write(f"CPU Cores: {system_info['cpu_count']}\n")
            f.write(f"CPU Max Freq: {system_info['cpu_freq_max']:.1f} MHz\n")
            f.write(f"Memory: {system_info['memory_total']:.1f} GB total, {system_info['memory_available']:.1f} GB available\n")
            f.write(f"Python: {system_info['python_version']}\n")
            f.write(f"PyTorch: {system_info['torch_version']}\n")
            f.write(f"Device: {system_info['device_name']} ({system_info['device_type']})\n")
            if system_info.get('cuda_version'):
                f.write(f"CUDA: {system_info['cuda_version']}\n")
            if system_info.get('mps_available'):
                f.write(f"MPS Available: {system_info['mps_available']}\n")
            f.write("\n")
            
            # Performance summary
            averages = metrics['averages_per_episode']
            f.write("PERFORMANCE SUMMARY (per episode)\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Episode Time: {averages.get('episode_total', 0):.3f}s\n\n")
            
            # Environment operations
            f.write("Environment Operations:\n")
            f.write(f"  Reset: {averages.get('env_reset', 0):.3f}s ({self._percentage(averages, 'env_reset', 'episode_total'):.1f}%)\n")
            f.write(f"  Step: {averages.get('env_step', 0):.3f}s ({self._percentage(averages, 'env_step', 'episode_total'):.1f}%)\n")
            f.write(f"  Close: {averages.get('env_close', 0):.3f}s ({self._percentage(averages, 'env_close', 'episode_total'):.1f}%)\n\n")
            
            # Battle operations
            f.write("Battle Operations:\n")
            f.write(f"  Initialization: {averages.get('battle_init', 0):.3f}s ({self._percentage(averages, 'battle_init', 'episode_total'):.1f}%)\n")
            f.write(f"  Progress: {averages.get('battle_progress', 0):.3f}s ({self._percentage(averages, 'battle_progress', 'episode_total'):.1f}%)\n")
            f.write(f"  WebSocket: {averages.get('battle_websocket', 0):.3f}s ({self._percentage(averages, 'battle_websocket', 'episode_total'):.1f}%)\n\n")
            
            # Agent operations
            f.write("Agent Operations:\n")
            f.write(f"  Action Selection: {averages.get('agent_action_selection', 0):.3f}s ({self._percentage(averages, 'agent_action_selection', 'episode_total'):.1f}%)\n")
            f.write(f"  Value Calculation: {averages.get('agent_value_calculation', 0):.3f}s ({self._percentage(averages, 'agent_value_calculation', 'episode_total'):.1f}%)\n\n")
            
            # Learning operations
            f.write("Learning Operations:\n")
            f.write(f"  Gradient Calculation: {averages.get('gradient_calculation', 0):.3f}s ({self._percentage(averages, 'gradient_calculation', 'episode_total'):.1f}%)\n")
            f.write(f"  Optimizer Step: {averages.get('optimizer_step', 0):.3f}s ({self._percentage(averages, 'optimizer_step', 'episode_total'):.1f}%)\n")
            f.write(f"  Loss Calculation: {averages.get('loss_calculation', 0):.3f}s ({self._percentage(averages, 'loss_calculation', 'episode_total'):.1f}%)\n\n")
            
            # I/O operations
            f.write("I/O Operations:\n")
            f.write(f"  Model Save: {averages.get('model_save', 0):.3f}s ({self._percentage(averages, 'model_save', 'episode_total'):.1f}%)\n")
            f.write(f"  Model Load: {averages.get('model_load', 0):.3f}s ({self._percentage(averages, 'model_load', 'episode_total'):.1f}%)\n")
            f.write(f"  Data Logging: {averages.get('data_logging', 0):.3f}s ({self._percentage(averages, 'data_logging', 'episode_total'):.1f}%)\n\n")
            
            # Memory operations
            f.write("Memory Operations:\n")
            f.write(f"  Tensor Operations: {averages.get('tensor_operations', 0):.3f}s ({self._percentage(averages, 'tensor_operations', 'episode_total'):.1f}%)\n")
            f.write(f"  Device Transfer: {averages.get('device_transfer', 0):.3f}s ({self._percentage(averages, 'device_transfer', 'episode_total'):.1f}%)\n\n")
            
            # Other operations
            f.write("Other Operations:\n")
            f.write(f"  Reward Calculation: {averages.get('reward_calculation', 0):.3f}s ({self._percentage(averages, 'reward_calculation', 'episode_total'):.1f}%)\n")
            f.write(f"  State Observation: {averages.get('state_observation', 0):.3f}s ({self._percentage(averages, 'state_observation', 'episode_total'):.1f}%)\n")
            f.write(f"  Action Masking: {averages.get('action_masking', 0):.3f}s ({self._percentage(averages, 'action_masking', 'episode_total'):.1f}%)\n\n")
            
            # System metrics
            system_metrics = metrics['system_metrics']
            f.write("SYSTEM RESOURCE USAGE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Peak Memory Usage: {system_metrics['memory_usage_peak']:.2f} GB\n")
            f.write(f"Average CPU Usage: {system_metrics['cpu_usage_avg']:.1f}%\n")
            f.write(f"Average GPU Usage: {system_metrics['gpu_usage_avg']:.1f}%\n\n")
            
            # Metadata
            if data.get('metadata'):
                f.write("METADATA\n")
                f.write("-" * 20 + "\n")
                for key, value in data['metadata'].items():
                    f.write(f"{key}: {value}\n")
    
    def _percentage(self, averages: Dict[str, float], part: str, total: str) -> float:
        """Calculate percentage of part relative to total."""
        total_val = averages.get(total, 0)
        part_val = averages.get(part, 0)
        if total_val == 0:
            return 0.0
        return (part_val / total_val) * 100
    
    def create_comparison_report(self, session_files: List[Path], output_name: str) -> Path:
        """Create a comparison report from multiple profiling sessions."""
        if not self.enabled:
            return Path()
        
        comparison_file = self.log_dir / "comparison" / f"{output_name}_comparison.csv"
        
        # Load all sessions
        sessions = []
        for file_path in session_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    sessions.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        if not sessions:
            return comparison_file
        
        # Create CSV comparison
        with open(comparison_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Metric']
            for session in sessions:
                session_name = session.get('session_name', 'Unknown')
                timestamp = session.get('timestamp', '')
                header.append(f"{session_name}_{timestamp}")
            writer.writerow(header)
            
            # System info comparison
            writer.writerow(['=== SYSTEM INFO ==='])
            system_keys = ['platform', 'processor', 'cpu_count', 'memory_total', 'device_name', 'device_type']
            for key in system_keys:
                row = [key.replace('_', ' ').title()]
                for session in sessions:
                    value = session.get('system_info', {}).get(key, 'N/A')
                    row.append(str(value))
                writer.writerow(row)
            
            writer.writerow([])  # Empty row
            
            # Performance comparison
            writer.writerow(['=== PERFORMANCE (per episode) ==='])
            perf_keys = [
                'episode_total', 'env_reset', 'env_step', 'battle_init', 'battle_progress',
                'agent_action_selection', 'gradient_calculation', 'optimizer_step',
                'tensor_operations', 'reward_calculation'
            ]
            
            for key in perf_keys:
                row = [key.replace('_', ' ').title() + ' (s)']
                for session in sessions:
                    averages = session.get('metrics', {}).get('averages_per_episode', {})
                    value = averages.get(key, 0)
                    row.append(f"{value:.4f}")
                writer.writerow(row)
            
            writer.writerow([])  # Empty row
            
            # System usage comparison
            writer.writerow(['=== RESOURCE USAGE ==='])
            usage_keys = ['memory_usage_peak', 'cpu_usage_avg', 'gpu_usage_avg']
            units = [' (GB)', ' (%)', ' (%)']
            
            for key, unit in zip(usage_keys, units):
                row = [key.replace('_', ' ').title() + unit]
                for session in sessions:
                    system_metrics = session.get('metrics', {}).get('system_metrics', {})
                    value = system_metrics.get(key, 0)
                    row.append(f"{value:.2f}")
                writer.writerow(row)
        
        return comparison_file
    
    def get_latest_sessions(self, count: int = 10) -> List[Path]:
        """Get the most recent profiling session files."""
        if not self.enabled:
            return []
        
        raw_dir = self.log_dir / "raw"
        if not raw_dir.exists():
            return []
        
        # Get all JSON files sorted by modification time
        json_files = list(raw_dir.glob("*.json"))
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return json_files[:count]