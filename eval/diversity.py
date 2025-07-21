"""
行動多様性メトリクス

V-3タスク: 行動多様性メトリクス
技選択分布のKL距離を算出しグラフ化する機能を提供します。

使用例:
    from eval.diversity import ActionDiversityAnalyzer, calculate_action_kl_divergence
    
    # 行動履歴から多様性分析
    analyzer = ActionDiversityAnalyzer()
    action_sequence = [0, 1, 2, 0, 3, 1, 0]
    diversity = analyzer.calculate_diversity(action_sequence)
    
    # KL距離計算
    old_dist = [0.6, 0.3, 0.1]
    new_dist = [0.4, 0.4, 0.2]
    kl_div = calculate_action_kl_divergence(old_dist, new_dist)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
from pathlib import Path
# Optional imports for advanced analysis
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from scipy.stats import entropy
    from scipy.spatial.distance import jensenshannon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ActionDiversityAnalyzer:
    """
    行動多様性の分析クラス
    
    行動序列から以下のメトリクスを算出:
    - エントロピー（情報エントロピー）
    - ジニ係数（均等性指標）
    - シャノン多様性指数
    - 行動分布のKL距離
    """
    
    def __init__(self, num_actions: int = 11):
        """
        Args:
            num_actions: 行動の総数（ポケモンの場合11: 技4種+テラス4種+交代2種+struggle）
        """
        self.num_actions = num_actions
        self.action_history: List[int] = []
        self.episode_distributions: List[np.ndarray] = []
        
    def add_episode_actions(self, actions: List[int]) -> None:
        """
        エピソードの行動履歴を追加
        
        Args:
            actions: 行動IDのリスト
        """
        self.action_history.extend(actions)
        
        # エピソード内行動分布を計算・保存
        action_counts = Counter(actions)
        distribution = np.zeros(self.num_actions)
        
        for action_id, count in action_counts.items():
            if 0 <= action_id < self.num_actions:
                distribution[action_id] = count
                
        # 正規化（確率分布化）
        if distribution.sum() > 0:
            distribution = distribution / distribution.sum()
            
        self.episode_distributions.append(distribution)
        
    def calculate_diversity(self, actions: Optional[List[int]] = None) -> Dict[str, float]:
        """
        行動多様性メトリクスを計算
        
        Args:
            actions: 行動リスト（省略時は蓄積された全履歴を使用）
            
        Returns:
            多様性メトリクス辞書
        """
        if actions is None:
            if not self.action_history:
                raise ValueError("行動履歴が空です。add_episode_actions()で行動を追加してください。")
            actions = self.action_history
            
        # 行動分布を計算
        action_counts = Counter(actions)
        distribution = np.zeros(self.num_actions)
        
        for action_id, count in action_counts.items():
            if 0 <= action_id < self.num_actions:
                distribution[action_id] = count
                
        if distribution.sum() == 0:
            return {
                'entropy': 0.0,
                'normalized_entropy': 0.0,
                'gini_coefficient': 1.0,
                'shannon_diversity': 0.0,
                'effective_actions': 0.0,
                'uniformity_index': 0.0
            }
        
        # 確率分布に正規化
        prob_distribution = distribution / distribution.sum()
        
        # 1. Shannon Entropy（情報エントロピー）
        if HAS_SCIPY:
            shannon_entropy = entropy(prob_distribution, base=2)
        else:
            # Manual entropy calculation
            shannon_entropy = -np.sum(prob_distribution * np.log2(prob_distribution + 1e-10))
            
        max_entropy = np.log2(self.num_actions)
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # 2. Gini係数（不平等性指標、0=完全平等, 1=完全不平等）
        gini = self._calculate_gini_coefficient(distribution)
        
        # 3. Shannon多様性指数（生態学指標）
        shannon_diversity = entropy(prob_distribution, base=np.e)
        
        # 4. 実効行動数（Hill数、指数化多様性）
        effective_actions = np.exp(shannon_diversity)
        
        # 5. 均等性指数（Pielou's evenness）
        max_possible_diversity = np.log(self.num_actions)
        uniformity_index = shannon_diversity / max_possible_diversity if max_possible_diversity > 0 else 0.0
        
        return {
            'entropy': shannon_entropy,
            'normalized_entropy': normalized_entropy,
            'gini_coefficient': gini,
            'shannon_diversity': shannon_diversity,
            'effective_actions': effective_actions,
            'uniformity_index': uniformity_index
        }
        
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Gini係数を計算（内部メソッド）
        
        Args:
            values: 数値配列
            
        Returns:
            Gini係数（0-1、0が完全平等）
        """
        if len(values) == 0 or np.sum(values) == 0:
            return 0.0
            
        # ソートして累積和計算
        sorted_values = np.sort(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        # Gini係数の計算式
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
        return max(0.0, gini)  # 負値を0でクリップ
        
    def calculate_kl_divergence_timeline(self, window_size: int = 50) -> List[float]:
        """
        時系列でのKL距離変化を計算
        
        Args:
            window_size: 移動窓サイズ
            
        Returns:
            KL距離の時系列リスト
        """
        if len(self.episode_distributions) < 2:
            return []
        
        kl_divergences = []
        
        for i in range(1, len(self.episode_distributions)):
            prev_dist = self.episode_distributions[i-1]
            curr_dist = self.episode_distributions[i]
            
            # 数値的安定化（小さな値を追加）
            prev_dist_stable = prev_dist + 1e-10
            curr_dist_stable = curr_dist + 1e-10
            
            # KL距離計算（対称化）
            if HAS_SCIPY:
                kl_div = 0.5 * (entropy(prev_dist_stable, curr_dist_stable) + 
                               entropy(curr_dist_stable, prev_dist_stable))
            else:
                # Manual KL divergence calculation
                kl1 = np.sum(prev_dist_stable * np.log(prev_dist_stable / curr_dist_stable))
                kl2 = np.sum(curr_dist_stable * np.log(curr_dist_stable / prev_dist_stable))
                kl_div = 0.5 * (kl1 + kl2)
            kl_divergences.append(kl_div)
        
        return kl_divergences
        
    def plot_action_distribution(
        self, 
        output_path: Optional[str] = None,
        action_names: Optional[List[str]] = None
    ) -> str:
        """
        行動分布をプロット
        
        Args:
            output_path: 出力ファイルパス（省略時は自動生成）
            action_names: 行動名リスト
            
        Returns:
            作成されたプロットファイルのパス
        """
        if not self.action_history:
            raise ValueError("行動履歴が空です。")
        
        # デフォルト行動名
        if action_names is None:
            action_names = [
                "Move 1", "Move 2", "Move 3", "Move 4",
                "Tera Move 1", "Tera Move 2", "Tera Move 3", "Tera Move 4",
                "Switch 1", "Switch 2", "Struggle"
            ][:self.num_actions]
        
        # 行動分布を計算
        action_counts = Counter(self.action_history)
        actions = list(range(self.num_actions))
        counts = [action_counts.get(i, 0) for i in actions]
        
        # プロット作成
        plt.figure(figsize=(12, 6))
        bars = plt.bar(actions, counts, alpha=0.7, color='skyblue', edgecolor='navy')
        
        plt.title('Action Distribution Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Action Type', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(actions, [f"{i}\n{name}" for i, name in enumerate(action_names)], rotation=45)
        
        # 値をバーの上に表示
        for bar, count in zip(bars, counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 出力パス決定
        if output_path is None:
            import time
            output_path = f"action_distribution_{int(time.time())}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"行動分布グラフを保存しました: {output_path}")
        return output_path
        
    def plot_diversity_timeline(
        self, 
        output_path: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> str:
        """
        多様性メトリクスの時系列プロット
        
        Args:
            output_path: 出力ファイルパス（省略時は自動生成）
            metrics: プロットするメトリクスリスト
            
        Returns:
            作成されたプロットファイルのパス
        """
        if len(self.episode_distributions) < 2:
            raise ValueError("エピソード分布が2つ未満です。")
        
        if metrics is None:
            metrics = ['entropy', 'gini_coefficient', 'effective_actions']
        
        # 各エピソードの多様性メトリクスを計算
        timeline_metrics = {metric: [] for metric in metrics}
        episodes = []
        
        for i, dist in enumerate(self.episode_distributions):
            if dist.sum() == 0:
                continue
            
            # 仮の行動列を作成（分布から復元）
            actions = []
            for action_id, prob in enumerate(dist):
                if prob > 0:
                    actions.extend([action_id] * int(prob * 100))  # スケール調整
            
            if actions:
                diversity = self.calculate_diversity(actions)
                episodes.append(i + 1)
                for metric in metrics:
                    timeline_metrics[metric].append(diversity.get(metric, 0.0))
        
        if not episodes:
            raise ValueError("有効なエピソードデータがありません。")
        
        # プロット作成
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            axes[i].plot(episodes, timeline_metrics[metric], 'o-', linewidth=2, markersize=4)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Over Episodes', fontweight='bold')
            axes[i].set_xlabel('Episode')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 出力パス決定
        if output_path is None:
            import time
            output_path = f"diversity_timeline_{int(time.time())}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"多様性時系列グラフを保存しました: {output_path}")
        return output_path


def calculate_action_kl_divergence(
    dist1: List[float] | np.ndarray, 
    dist2: List[float] | np.ndarray,
    symmetric: bool = True
) -> float:
    """
    2つの行動分布間のKL距離を計算
    
    Args:
        dist1: 分布1（確率分布）
        dist2: 分布2（確率分布）  
        symmetric: 対称KL距離を使うかどうか
        
    Returns:
        KL距離
    """
    dist1 = np.asarray(dist1, dtype=float)
    dist2 = np.asarray(dist2, dtype=float)
    
    # 正規化
    if dist1.sum() > 0:
        dist1 = dist1 / dist1.sum()
    if dist2.sum() > 0:
        dist2 = dist2 / dist2.sum()
    
    # 数値的安定化
    dist1_stable = dist1 + 1e-10
    dist2_stable = dist2 + 1e-10
    
    if HAS_SCIPY:
        if symmetric:
            # 対称KL距離（Jensen-Shannon距離の代替）
            kl_div = 0.5 * (entropy(dist1_stable, dist2_stable) + entropy(dist2_stable, dist1_stable))
        else:
            # 標準KL距離
            kl_div = entropy(dist1_stable, dist2_stable)
    else:
        # Manual KL divergence calculation
        if symmetric:
            kl1 = np.sum(dist1_stable * np.log(dist1_stable / dist2_stable))
            kl2 = np.sum(dist2_stable * np.log(dist2_stable / dist1_stable))
            kl_div = 0.5 * (kl1 + kl2)
        else:
            kl_div = np.sum(dist1_stable * np.log(dist1_stable / dist2_stable))
    
    return float(kl_div)


def calculate_jensen_shannon_distance(
    dist1: List[float] | np.ndarray,
    dist2: List[float] | np.ndarray
) -> float:
    """
    Jensen-Shannon距離を計算（対称な距離メトリクス）
    
    Args:
        dist1: 分布1
        dist2: 分布2
        
    Returns:
        Jensen-Shannon距離
    """
    dist1 = np.asarray(dist1, dtype=float)
    dist2 = np.asarray(dist2, dtype=float)
    
    # 正規化
    if dist1.sum() > 0:
        dist1 = dist1 / dist1.sum()
    if dist2.sum() > 0:
        dist2 = dist2 / dist2.sum()
    
    # Jensen-Shannon距離
    if HAS_SCIPY:
        js_distance = jensenshannon(dist1, dist2)
    else:
        # Manual Jensen-Shannon distance calculation
        # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5 * (P + Q)
        m_dist = 0.5 * (dist1 + dist2)
        m_dist_stable = m_dist + 1e-10
        dist1_stable = dist1 + 1e-10
        dist2_stable = dist2 + 1e-10
        
        kl1 = np.sum(dist1_stable * np.log(dist1_stable / m_dist_stable))
        kl2 = np.sum(dist2_stable * np.log(dist2_stable / m_dist_stable))
        js_divergence = 0.5 * kl1 + 0.5 * kl2
        js_distance = np.sqrt(js_divergence)  # 距離に変換
        
    return float(js_distance)


def analyze_move_selection_patterns(
    move_sequences: List[List[int]],
    move_names: Optional[List[str]] = None,
    output_dir: str = "diversity_analysis"
) -> Dict[str, Any]:
    """
    技選択パターンの包括的分析
    
    Args:
        move_sequences: エピソード別の技選択履歴リスト
        move_names: 技名リスト
        output_dir: 出力ディレクトリ
        
    Returns:
        分析結果辞書
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 分析器を初期化
    analyzer = ActionDiversityAnalyzer(num_actions=11)  # ポケモンの行動数
    
    # データ追加
    for episode_moves in move_sequences:
        analyzer.add_episode_actions(episode_moves)
    
    # 多様性分析
    overall_diversity = analyzer.calculate_diversity()
    
    # グラフ作成
    dist_plot_path = analyzer.plot_action_distribution(
        output_path / "move_distribution.png",
        action_names=move_names
    )
    
    timeline_plot_path = analyzer.plot_diversity_timeline(
        output_path / "diversity_timeline.png"
    )
    
    # KL距離時系列
    kl_timeline = analyzer.calculate_kl_divergence_timeline()
    
    # 結果まとめ
    analysis_results = {
        'overall_diversity': overall_diversity,
        'kl_divergence_timeline': kl_timeline,
        'plots': {
            'distribution': dist_plot_path,
            'timeline': timeline_plot_path
        },
        'episode_count': len(move_sequences),
        'total_actions': len(analyzer.action_history)
    }
    
    # レポート作成
    report_path = output_path / "analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 技選択パターン分析レポート ===\n\n")
        f.write(f"分析エピソード数: {analysis_results['episode_count']}\n")
        f.write(f"総行動数: {analysis_results['total_actions']}\n\n")
        
        f.write("【多様性メトリクス】\n")
        for metric, value in overall_diversity.items():
            f.write(f"  {metric}: {value:.4f}\n")
        
        f.write(f"\n【KL距離統計】\n")
        if kl_timeline:
            f.write(f"  平均KL距離: {np.mean(kl_timeline):.4f}\n")
            f.write(f"  最大KL距離: {np.max(kl_timeline):.4f}\n")
            f.write(f"  最小KL距離: {np.min(kl_timeline):.4f}\n")
        
        f.write(f"\n【作成ファイル】\n")
        f.write(f"  行動分布グラフ: {Path(dist_plot_path).name}\n")
        f.write(f"  多様性時系列: {Path(timeline_plot_path).name}\n")
    
    analysis_results['report_path'] = str(report_path)
    
    print(f"技選択パターン分析が完了しました。結果は {output_dir} ディレクトリに保存されました。")
    return analysis_results


__all__ = [
    'ActionDiversityAnalyzer',
    'calculate_action_kl_divergence', 
    'calculate_jensen_shannon_distance',
    'analyze_move_selection_patterns'
]