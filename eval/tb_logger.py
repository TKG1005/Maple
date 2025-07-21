"""
TensorBoard統一ロガーモジュール

V-1タスク: TensorBoard スカラー整理
統一命名規則でのメトリクス記録を提供します。

使用例:
    from eval.tb_logger import TensorBoardLogger
    
    logger = TensorBoardLogger(log_dir="runs/experiment1")
    logger.log_training_metrics(episode=1, loss=0.5, entropy=1.2)
    logger.log_reward_metrics(episode=1, total_reward=10.0, sub_rewards={'knockout': 5.0})
    logger.log_exploration_metrics(episode=1, epsilon=0.8, random_actions=45)
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class TensorBoardLogger:
    """
    統一命名規則でTensorBoardメトリクスを記録するロガー
    
    命名規則:
    - Training: loss, entropy, learning_rate
    - Rewards: reward/total, reward/avg, reward/{sub_reward_name}
    - Performance: win_rate, episode_duration  
    - Exploration: exploration/{metric_name}
    - Diversity: diversity/{metric_name}
    """
    
    # 統一命名規則の定義
    METRIC_NAMES = {
        # Training metrics
        'loss': 'training/loss',
        'entropy': 'training/entropy', 
        'entropy_avg': 'training/entropy_avg',
        'learning_rate': 'training/learning_rate',
        
        # Reward metrics
        'total_reward': 'reward/total',
        'avg_reward': 'reward/average',
        'win_rate': 'performance/win_rate',
        'episode_duration': 'performance/episode_duration',
        'time_per_episode': 'performance/time_per_episode',
        
        # Exploration metrics (ε-greedy)
        'epsilon': 'exploration/epsilon',
        'random_actions': 'exploration/random_actions',
        'total_actions': 'exploration/total_actions', 
        'exploration_rate': 'exploration/exploration_rate',
        'random_action_rate': 'exploration/random_action_rate',
        'decay_progress': 'exploration/decay_progress',
        'epsilon_start': 'exploration/epsilon_start',
        'epsilon_end': 'exploration/epsilon_end',
        'episode_count': 'exploration/episode_count',
        'step_count': 'exploration/step_count',
    }
    
    def __init__(self, log_dir: str | Path, experiment_name: Optional[str] = None):
        """
        TensorBoardLoggerを初期化
        
        Args:
            log_dir: ログディレクトリのパス
            experiment_name: 実験名（省略時は現在時刻を使用）
        """
        if experiment_name is None:
            experiment_name = f"experiment_{int(time.time())}"
            
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        
        # メトリクス履歴（CSVエクスポート用）
        self.metrics_history: Dict[str, list] = {}
        
    def _add_scalar(self, metric_key: str, value: float, step: int) -> None:
        """
        統一命名規則でスカラー値を記録（内部メソッド）
        
        Args:
            metric_key: メトリクスキー（METRIC_NAMESに定義）
            value: 記録する値
            step: ステップ番号（エピソード数など）
        """
        # 統一命名規則に基づく名前取得
        tensorboard_name = self.METRIC_NAMES.get(metric_key, metric_key)
        
        # TensorBoardに記録
        self.writer.add_scalar(tensorboard_name, value, step)
        
        # 履歴に記録（CSVエクスポート用）
        if tensorboard_name not in self.metrics_history:
            self.metrics_history[tensorboard_name] = []
        self.metrics_history[tensorboard_name].append((step, value))
        
    def log_training_metrics(
        self, 
        episode: int, 
        loss: Optional[float] = None,
        entropy: Optional[float] = None,
        entropy_avg: Optional[float] = None,
        learning_rate: Optional[float] = None
    ) -> None:
        """
        学習メトリクスを記録
        
        Args:
            episode: エピソード番号
            loss: 損失値
            entropy: エントロピー値
            entropy_avg: 平均エントロピー値
            learning_rate: 学習率
        """
        if loss is not None:
            self._add_scalar('loss', loss, episode)
        if entropy is not None:
            self._add_scalar('entropy', entropy, episode)
        if entropy_avg is not None:
            self._add_scalar('entropy_avg', entropy_avg, episode)
        if learning_rate is not None:
            self._add_scalar('learning_rate', learning_rate, episode)
            
    def log_reward_metrics(
        self,
        episode: int,
        total_reward: Optional[float] = None,
        avg_reward: Optional[float] = None,
        sub_rewards: Optional[Dict[str, float]] = None
    ) -> None:
        """
        報酬メトリクスを記録
        
        Args:
            episode: エピソード番号  
            total_reward: 総報酬
            avg_reward: 平均報酬
            sub_rewards: サブ報酬の辞書 {"knockout": 5.0, "turn_penalty": -1.0}
        """
        if total_reward is not None:
            self._add_scalar('total_reward', total_reward, episode)
        if avg_reward is not None:
            self._add_scalar('avg_reward', avg_reward, episode)
            
        # サブ報酬を個別記録
        if sub_rewards:
            for reward_name, reward_value in sub_rewards.items():
                sub_reward_key = f"reward/{reward_name}"
                self.writer.add_scalar(sub_reward_key, reward_value, episode)
                
                # 履歴に記録
                if sub_reward_key not in self.metrics_history:
                    self.metrics_history[sub_reward_key] = []
                self.metrics_history[sub_reward_key].append((episode, reward_value))
                
    def log_performance_metrics(
        self,
        episode: int,
        win_rate: Optional[float] = None,
        episode_duration: Optional[float] = None
    ) -> None:
        """
        パフォーマンスメトリクスを記録
        
        Args:
            episode: エピソード番号
            win_rate: 勝率
            episode_duration: エピソード実行時間
        """
        if win_rate is not None:
            self._add_scalar('win_rate', win_rate, episode)
        if episode_duration is not None:
            self._add_scalar('episode_duration', episode_duration, episode)
            
    def log_exploration_metrics(
        self,
        episode: int,
        exploration_stats: Dict[str, Any]
    ) -> None:
        """
        探索メトリクスを記録（ε-greedy統計）
        
        Args:
            episode: エピソード番号
            exploration_stats: 探索統計辞書（EpsilonGreedyWrapper.get_exploration_stats()の結果）
        """
        # 探索関連メトリクスのマッピング
        exploration_keys = [
            'epsilon', 'random_actions', 'total_actions',
            'exploration_rate', 'random_action_rate', 'decay_progress',
            'epsilon_start', 'epsilon_end', 'episode_count', 'step_count'
        ]
        
        for key in exploration_keys:
            if key in exploration_stats:
                self._add_scalar(key, float(exploration_stats[key]), episode)
                
    def log_diversity_metrics(
        self,
        episode: int,
        action_entropy: Optional[float] = None,
        move_diversity: Optional[float] = None,
        kl_divergence: Optional[float] = None
    ) -> None:
        """
        行動多様性メトリクスを記録
        
        Args:
            episode: エピソード番号
            action_entropy: 行動選択のエントロピー
            move_diversity: 技選択の多様性
            kl_divergence: 行動分布のKL距離
        """
        if action_entropy is not None:
            self.writer.add_scalar('diversity/action_entropy', action_entropy, episode)
        if move_diversity is not None:
            self.writer.add_scalar('diversity/move_diversity', move_diversity, episode)
        if kl_divergence is not None:
            self.writer.add_scalar('diversity/kl_divergence', kl_divergence, episode)
            
        # 履歴記録
        metrics = {
            'diversity/action_entropy': action_entropy,
            'diversity/move_diversity': move_diversity,
            'diversity/kl_divergence': kl_divergence
        }
        
        for key, value in metrics.items():
            if value is not None:
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append((episode, value))
    
    def get_metrics_history(self) -> Dict[str, list]:
        """
        記録されたメトリクス履歴を取得（CSVエクスポート用）
        
        Returns:
            メトリクス履歴辞書 {metric_name: [(step, value), ...]}
        """
        return self.metrics_history.copy()
        
    def flush(self) -> None:
        """TensorBoardバッファをフラッシュ"""
        self.writer.flush()
        
    def close(self) -> None:
        """ロガーを閉じる"""
        self.writer.close()
        
    def __enter__(self):
        """コンテキストマネージャー対応"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー対応"""
        self.close()


# 便利関数
def create_logger(experiment_name: str, base_dir: str = "runs") -> TensorBoardLogger:
    """
    TensorBoardLoggerを作成する便利関数
    
    Args:
        experiment_name: 実験名
        base_dir: ベースディレクトリ（デフォルト: "runs"）
        
    Returns:
        設定済みのTensorBoardLoggerインスタンス
    """
    return TensorBoardLogger(log_dir=base_dir, experiment_name=experiment_name)


__all__ = ['TensorBoardLogger', 'create_logger']