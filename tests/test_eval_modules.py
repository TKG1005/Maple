"""
評価モジュール（V1-V3）の包括的テスト

テスト対象:
- V1: TensorBoardLogger (tb_logger.py)
- V2: CSV export utility (export_csv.py)  
- V3: Action diversity metrics (diversity.py)
"""

from __future__ import annotations

import pytest
import tempfile
import numpy as np
import csv
from pathlib import Path
from unittest.mock import Mock, patch
import os
import shutil

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval.tb_logger import TensorBoardLogger, create_logger
from eval.export_csv import export_metrics_to_csv, create_experiment_summary, _convert_metrics_to_csv_format
from eval.diversity import (
    ActionDiversityAnalyzer, 
    calculate_action_kl_divergence, 
    calculate_jensen_shannon_distance,
    analyze_move_selection_patterns
)


class TestTensorBoardLogger:
    """V1: TensorBoardLoggerのテストクラス"""
    
    def setup_method(self):
        """各テストメソッド実行前の準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = TensorBoardLogger(log_dir=self.temp_dir, experiment_name="test_experiment")
        
    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        self.logger.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_logger_initialization(self):
        """ロガーの初期化テスト"""
        assert self.logger.log_dir.exists()
        assert self.logger.log_dir.name == "test_experiment"
        assert self.logger.writer is not None
        assert isinstance(self.logger.metrics_history, dict)
        
    def test_training_metrics_logging(self):
        """学習メトリクス記録テスト"""
        # 学習メトリクスを記録
        self.logger.log_training_metrics(
            episode=1,
            loss=0.5,
            entropy=1.2,
            entropy_avg=1.1,
            learning_rate=0.001
        )
        
        # 履歴に記録されているかチェック
        history = self.logger.get_metrics_history()
        assert 'training/loss' in history
        assert 'training/entropy' in history
        assert 'training/entropy_avg' in history
        assert 'training/learning_rate' in history
        
        # 値が正しく記録されているかチェック
        assert history['training/loss'][0] == (1, 0.5)
        assert history['training/entropy'][0] == (1, 1.2)
        
    def test_reward_metrics_logging(self):
        """報酬メトリクス記録テスト"""
        sub_rewards = {'knockout': 5.0, 'turn_penalty': -1.0}
        
        self.logger.log_reward_metrics(
            episode=2,
            total_reward=10.0,
            avg_reward=8.5,
            sub_rewards=sub_rewards
        )
        
        history = self.logger.get_metrics_history()
        assert 'reward/total' in history
        assert 'reward/average' in history
        assert 'reward/knockout' in history
        assert 'reward/turn_penalty' in history
        
        assert history['reward/total'][0] == (2, 10.0)
        assert history['reward/knockout'][0] == (2, 5.0)
        
    def test_performance_metrics_logging(self):
        """パフォーマンスメトリクス記録テスト"""
        self.logger.log_performance_metrics(
            episode=3,
            win_rate=0.75,
            episode_duration=120.5
        )
        
        history = self.logger.get_metrics_history()
        assert 'performance/win_rate' in history
        assert 'performance/episode_duration' in history
        
        assert history['performance/win_rate'][0] == (3, 0.75)
        
    def test_exploration_metrics_logging(self):
        """探索メトリクス記録テスト"""
        exploration_stats = {
            'epsilon': 0.8,
            'random_actions': 45,
            'total_actions': 67,
            'exploration_rate': 0.67,
            'random_action_rate': 0.67,
            'decay_progress': 0.1
        }
        
        self.logger.log_exploration_metrics(episode=4, exploration_stats=exploration_stats)
        
        history = self.logger.get_metrics_history()
        assert 'exploration/epsilon' in history
        assert 'exploration/random_actions' in history
        assert 'exploration/total_actions' in history
        
        assert history['exploration/epsilon'][0] == (4, 0.8)
        assert history['exploration/random_actions'][0] == (4, 45)
        
    def test_diversity_metrics_logging(self):
        """多様性メトリクス記録テスト"""
        self.logger.log_diversity_metrics(
            episode=5,
            action_entropy=2.3,
            move_diversity=0.85,
            kl_divergence=0.12
        )
        
        history = self.logger.get_metrics_history()
        assert 'diversity/action_entropy' in history
        assert 'diversity/move_diversity' in history
        assert 'diversity/kl_divergence' in history
        
        assert history['diversity/action_entropy'][0] == (5, 2.3)
        
    def test_context_manager(self):
        """コンテキストマネージャーテスト"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            with TensorBoardLogger(log_dir=temp_dir, experiment_name="context_test") as logger:
                logger.log_training_metrics(episode=1, loss=0.1)
                history = logger.get_metrics_history()
                assert 'training/loss' in history
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    def test_create_logger_convenience_function(self):
        """便利関数のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = create_logger("convenience_test", base_dir=temp_dir)
            logger.log_training_metrics(episode=1, loss=0.2)
            logger.close()
            
            assert (Path(temp_dir) / "convenience_test").exists()


class TestCSVExport:
    """V2: CSV export utilityのテストクラス"""
    
    def setup_method(self):
        """各テストメソッド実行前の準備"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """各テストメソッド実行後のクリーンアップ"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_convert_metrics_to_csv_format(self):
        """メトリクス変換テスト"""
        metrics_data = {
            'training/loss': [(1, 0.5), (2, 0.4), (3, 0.3)],
            'reward/total': [(1, 10.0), (2, 12.0), (3, 15.0)]
        }
        
        csv_data = _convert_metrics_to_csv_format(metrics_data, include_timestamp=False)
        
        assert 'headers' in csv_data
        assert 'rows' in csv_data
        assert 'episode' in csv_data['headers']
        assert 'training/loss' in csv_data['headers']
        assert 'reward/total' in csv_data['headers']
        
        # データ行数チェック
        assert len(csv_data['rows']) == 3
        
        # 最初の行チェック
        first_row = csv_data['rows'][0]
        assert first_row[0] == 1  # episode
        
        # メトリクスの順序をチェック（ソート順）
        headers = csv_data['headers']
        reward_index = headers.index('reward/total')
        loss_index = headers.index('training/loss')
        
        assert first_row[reward_index] == 10.0  # total_reward
        assert first_row[loss_index] == 0.5  # loss
        
    def test_export_metrics_to_csv(self):
        """CSVエクスポートテスト"""
        # TensorBoardLoggerを作成してデータ記録
        logger = TensorBoardLogger(log_dir=self.temp_dir, experiment_name="csv_test")
        
        logger.log_training_metrics(episode=1, loss=0.5, entropy=1.2)
        logger.log_training_metrics(episode=2, loss=0.4, entropy=1.1)
        logger.log_reward_metrics(episode=1, total_reward=10.0)
        logger.log_reward_metrics(episode=2, total_reward=12.0)
        
        # CSVエクスポート
        csv_path = Path(self.temp_dir) / "test_metrics.csv"
        result_path = export_metrics_to_csv(logger, str(csv_path), include_timestamp=False)
        
        assert Path(result_path).exists()
        assert str(csv_path) == result_path
        
        # CSVファイル内容チェック
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 2
        assert 'episode' in rows[0]
        assert 'training/loss' in rows[0]
        assert 'reward/total' in rows[0]
        
        assert float(rows[0]['training/loss']) == 0.5
        assert float(rows[1]['training/loss']) == 0.4
        
        logger.close()
        
    def test_create_experiment_summary(self):
        """実験サマリー作成テスト"""
        # テスト用CSVファイル作成
        csv_path = Path(self.temp_dir) / "test_data.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'training/loss', 'reward/total'])
            writer.writerow([1, 0.5, 10.0])
            writer.writerow([2, 0.4, 12.0])
            writer.writerow([3, 0.3, 15.0])
            
        # サマリー作成
        summary_path = create_experiment_summary(str(csv_path))
        
        assert Path(summary_path).exists()
        
        # サマリー内容チェック
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        assert '実験サマリー' in content
        assert 'training/loss' in content
        assert 'reward/total' in content
        assert '平均:' in content
        assert '最小:' in content
        assert '最大:' in content


class TestActionDiversityAnalyzer:
    """V3: Action diversity metricsのテストクラス"""
    
    def setup_method(self):
        """各テストメソッド実行前の準備"""
        self.analyzer = ActionDiversityAnalyzer(num_actions=4)  # テスト用簡易版
        
    def test_analyzer_initialization(self):
        """分析器の初期化テスト"""
        assert self.analyzer.num_actions == 4
        assert self.analyzer.action_history == []
        assert self.analyzer.episode_distributions == []
        
    def test_add_episode_actions(self):
        """エピソード行動追加テスト"""
        actions = [0, 1, 2, 0, 1, 0]
        self.analyzer.add_episode_actions(actions)
        
        assert self.analyzer.action_history == actions
        assert len(self.analyzer.episode_distributions) == 1
        
        # 分布チェック
        distribution = self.analyzer.episode_distributions[0]
        expected_dist = np.array([3/6, 2/6, 1/6, 0/6])  # 正規化された分布
        np.testing.assert_array_almost_equal(distribution, expected_dist)
        
    def test_calculate_diversity_uniform_distribution(self):
        """均等分布での多様性計算テスト"""
        # 完全に均等な行動分布
        actions = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        diversity = self.analyzer.calculate_diversity(actions)
        
        # エントロピーは最大に近い
        assert diversity['entropy'] > 1.8  # log2(4) = 2に近い
        assert diversity['normalized_entropy'] > 0.9  # 1に近い
        
        # Gini係数は0に近い（平等）
        assert diversity['gini_coefficient'] < 0.1
        
        # 実効行動数は4に近い
        assert diversity['effective_actions'] > 3.5
        
    def test_calculate_diversity_skewed_distribution(self):
        """偏った分布での多様性計算テスト"""
        # 1つの行動が支配的
        actions = [0] * 9 + [1]  # 90% action 0, 10% action 1
        diversity = self.analyzer.calculate_diversity(actions)
        
        # エントロピーは低い
        assert diversity['entropy'] < 0.5
        assert diversity['normalized_entropy'] < 0.3
        
        # Gini係数は高い（不平等）
        assert diversity['gini_coefficient'] > 0.6
        
        # 実効行動数は1に近い
        assert diversity['effective_actions'] < 2.0
        
    def test_calculate_diversity_empty_actions(self):
        """空の行動リストでの多様性計算テスト"""
        diversity = self.analyzer.calculate_diversity([])
        
        assert diversity['entropy'] == 0.0
        assert diversity['normalized_entropy'] == 0.0
        assert diversity['gini_coefficient'] == 1.0
        assert diversity['effective_actions'] == 0.0
        
    def test_kl_divergence_timeline(self):
        """KL距離時系列計算テスト"""
        # 複数エピソードを追加
        self.analyzer.add_episode_actions([0, 0, 1, 1])  # 均等
        self.analyzer.add_episode_actions([0, 0, 0, 1])  # 偏り
        self.analyzer.add_episode_actions([1, 1, 1, 0])  # 逆偏り
        
        kl_timeline = self.analyzer.calculate_kl_divergence_timeline()
        
        assert len(kl_timeline) == 2  # n-1個のKL距離
        assert all(kl >= 0 for kl in kl_timeline)  # KL距離は非負
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_action_distribution(self, mock_close, mock_savefig):
        """行動分布プロットテスト"""
        actions = [0, 1, 2, 0, 1, 0]
        self.analyzer.add_episode_actions(actions)
        
        plot_path = self.analyzer.plot_action_distribution()
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        assert plot_path.endswith('.png')
        
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_diversity_timeline(self, mock_close, mock_savefig):
        """多様性時系列プロットテスト"""
        # 複数エピソードデータ
        self.analyzer.add_episode_actions([0, 1, 2, 3])
        self.analyzer.add_episode_actions([0, 0, 1, 2])
        self.analyzer.add_episode_actions([0, 0, 0, 1])
        
        plot_path = self.analyzer.plot_diversity_timeline()
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        assert plot_path.endswith('.png')


class TestDiversityUtilityFunctions:
    """多様性ユーティリティ関数のテスト"""
    
    def test_calculate_action_kl_divergence(self):
        """行動KL距離計算テスト"""
        dist1 = [0.25, 0.25, 0.25, 0.25]  # 均等分布
        dist2 = [0.7, 0.1, 0.1, 0.1]     # 偏った分布
        
        # KL距離計算
        kl_div = calculate_action_kl_divergence(dist1, dist2)
        
        assert kl_div > 0  # KL距離は正の値
        assert isinstance(kl_div, float)
        
        # 同じ分布間のKL距離は0に近い
        kl_same = calculate_action_kl_divergence(dist1, dist1)
        assert kl_same < 0.01
        
    def test_calculate_action_kl_divergence_symmetric(self):
        """対称KL距離計算テスト"""
        dist1 = [0.8, 0.1, 0.05, 0.05]
        dist2 = [0.05, 0.05, 0.1, 0.8]
        
        # 対称KL距離
        kl_sym = calculate_action_kl_divergence(dist1, dist2, symmetric=True)
        kl_sym_reverse = calculate_action_kl_divergence(dist2, dist1, symmetric=True)
        
        # 対称性チェック（ほぼ等しい）
        assert abs(kl_sym - kl_sym_reverse) < 1e-6
        
    def test_calculate_jensen_shannon_distance(self):
        """Jensen-Shannon距離計算テスト"""
        dist1 = [0.5, 0.3, 0.2]
        dist2 = [0.2, 0.3, 0.5]
        
        js_dist = calculate_jensen_shannon_distance(dist1, dist2)
        
        assert 0 <= js_dist <= 1  # JS距離は0-1の範囲
        assert isinstance(js_dist, float)
        
        # 同じ分布間のJS距離は0
        js_same = calculate_jensen_shannon_distance(dist1, dist1)
        assert js_same < 1e-10
        
    @patch('eval.diversity.ActionDiversityAnalyzer.plot_action_distribution')
    @patch('eval.diversity.ActionDiversityAnalyzer.plot_diversity_timeline')
    def test_analyze_move_selection_patterns(self, mock_timeline, mock_dist):
        """技選択パターン包括分析テスト"""
        # モックの戻り値設定
        mock_dist.return_value = "test_dist.png"
        mock_timeline.return_value = "test_timeline.png"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            move_sequences = [
                [0, 1, 2, 0, 1],
                [1, 1, 2, 3, 0],
                [0, 0, 1, 2, 2]
            ]
            
            results = analyze_move_selection_patterns(
                move_sequences, 
                output_dir=temp_dir
            )
            
            # 結果構造チェック
            assert 'overall_diversity' in results
            assert 'kl_divergence_timeline' in results
            assert 'plots' in results
            assert 'episode_count' in results
            assert 'total_actions' in results
            
            assert results['episode_count'] == 3
            assert results['total_actions'] == 15
            
            # レポートファイル作成チェック
            assert 'report_path' in results
            assert Path(results['report_path']).exists()


@pytest.mark.integration
class TestIntegration:
    """統合テスト"""
    
    def test_full_pipeline_integration(self):
        """全パイプライン統合テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. TensorBoardLoggerでデータ記録
            logger = TensorBoardLogger(log_dir=temp_dir, experiment_name="integration_test")
            
            # 複数エピソードのデータを記録
            for episode in range(1, 6):
                logger.log_training_metrics(
                    episode=episode,
                    loss=0.5 - episode * 0.05,  # 減少傾向
                    entropy=1.0 + episode * 0.1  # 増加傾向
                )
                logger.log_reward_metrics(
                    episode=episode,
                    total_reward=episode * 2.0,
                    sub_rewards={'knockout': episode * 1.5, 'penalty': -episode * 0.5}
                )
                
                # 探索統計（模擬データ）
                exploration_stats = {
                    'epsilon': 1.0 - episode * 0.15,
                    'random_actions': max(0, 50 - episode * 8),
                    'total_actions': 60,
                    'exploration_rate': max(0, 0.8 - episode * 0.15)
                }
                logger.log_exploration_metrics(episode, exploration_stats)
                
                # 多様性メトリクス（模擬データ）
                logger.log_diversity_metrics(
                    episode=episode,
                    action_entropy=2.0 - episode * 0.2,
                    kl_divergence=0.1 + episode * 0.02
                )
            
            # 2. CSVエクスポート
            csv_path = export_metrics_to_csv(logger, include_timestamp=False)
            assert Path(csv_path).exists()
            
            # 3. サマリー作成
            summary_path = create_experiment_summary(csv_path)
            assert Path(summary_path).exists()
            
            # 4. 行動多様性分析
            analyzer = ActionDiversityAnalyzer(num_actions=4)
            
            # 模擬行動データ
            move_sequences = [
                [0, 1, 2, 0],
                [1, 2, 3, 1], 
                [0, 0, 1, 2],
                [2, 3, 0, 1],
                [1, 1, 2, 3]
            ]
            
            for actions in move_sequences:
                analyzer.add_episode_actions(actions)
            
            diversity = analyzer.calculate_diversity()
            kl_timeline = analyzer.calculate_kl_divergence_timeline()
            
            # 結果検証
            assert 'entropy' in diversity
            assert len(kl_timeline) == len(move_sequences) - 1
            
            logger.close()
            
            print("統合テストが正常に完了しました！")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])