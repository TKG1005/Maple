#!/usr/bin/env python3
"""プロファイリング問題の修正スクリプト"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

def analyze_profiling_issues():
    """プロファイリング問題を分析"""
    print("🔍 プロファイリング問題を分析中...")
    
    # 最新のプロファイリングファイルを読み込み
    import json
    profiling_dir = Path("logs/profiling/raw")
    
    if not profiling_dir.exists():
        print("❌ プロファイリングディレクトリが存在しません")
        return
    
    json_files = list(profiling_dir.glob("*.json"))
    if not json_files:
        print("❌ プロファイリングファイルが見つかりません")
        return
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"📊 分析対象: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    metrics = data.get('metrics', {})
    totals = metrics.get('totals', {})
    averages = metrics.get('averages_per_episode', {})
    episode_count = metrics.get('episode_count', 0)
    
    print(f"\n📈 基本統計:")
    print(f"  エピソード数: {episode_count}")
    print(f"  総エピソード時間: {totals.get('episode_total', 0):.3f}s")
    print(f"  平均エピソード時間: {averages.get('episode_total', 0):.3f}s")
    
    print(f"\n⚠️ 問題の分析:")
    
    # 1. 時間の整合性チェック
    episode_total = totals.get('episode_total', 0)
    env_step_total = totals.get('env_step', 0)
    
    if env_step_total > episode_total * 2:
        print(f"❌ env_step時間が異常: {env_step_total:.3f}s > {episode_total:.3f}s")
        print("   原因: ThreadPoolExecutorによる重複カウント")
    
    # 2. 未計測の操作を確認
    zero_operations = []
    for key, value in totals.items():
        if value == 0 and key not in ['env_close', 'battle_init', 'battle_progress', 'battle_websocket']:
            zero_operations.append(key)
    
    if zero_operations:
        print(f"❌ 計測されていない操作: {', '.join(zero_operations)}")
    
    # 3. システムメトリクス確認
    system_metrics = metrics.get('system_metrics', {})
    cpu_usage = system_metrics.get('cpu_usage_avg', 0)
    gpu_usage = system_metrics.get('gpu_usage_avg', 0)
    
    if cpu_usage == 0:
        print(f"⚠️ CPU使用率が0%: モニタリング問題")
    if gpu_usage == 0:
        print(f"⚠️ GPU使用率が0%: 正常（CPU使用のため）")
    
    return {
        'episode_count': episode_count,
        'total_time': episode_total,
        'avg_time': averages.get('episode_total', 0),
        'threading_issue': env_step_total > episode_total * 2,
        'zero_operations': zero_operations,
        'cpu_monitoring': cpu_usage > 0
    }

def identify_fixes_needed():
    """必要な修正を特定"""
    analysis = analyze_profiling_issues()
    
    if not analysis:
        return
    
    print(f"\n🔧 必要な修正:")
    
    fixes = []
    
    if analysis['threading_issue']:
        fixes.append("1. ThreadPoolExecutor問題: プロファイラーをメイン関数でのみ実行")
        fixes.append("   - エピソード関数でのプロファイリングを削除")
        fixes.append("   - メイン関数でエピソード全体の時間を計測")
    
    if analysis['zero_operations']:
        fixes.append("2. 学習フェーズのプロファイリング追加:")
        for op in analysis['zero_operations']:
            if 'gradient' in op or 'optimizer' in op or 'loss' in op:
                fixes.append(f"   - {op}の計測追加")
    
    if not analysis['cpu_monitoring']:
        fixes.append("3. CPU使用率モニタリング修正")
    
    for i, fix in enumerate(fixes, 1):
        print(f"  {fix}")
    
    return fixes

def main():
    """メイン実行関数"""
    print("🔍 プロファイリング問題の診断開始")
    print("=" * 50)
    
    analysis = analyze_profiling_issues()
    fixes = identify_fixes_needed()
    
    print(f"\n💡 修正優先順位:")
    print("1. 最優先: ThreadPoolExecutor問題（時間計測の正確性）")
    print("2. 中優先: 学習フェーズプロファイリング（完全性）") 
    print("3. 低優先: CPU使用率モニタリング（システム情報）")
    
    print(f"\n次のステップ:")
    print("- train.pyのプロファイリング実装を修正")
    print("- ThreadPoolExecutorでの重複計測を回避")
    print("- 学習ループにプロファイリングを追加")

if __name__ == "__main__":
    main()