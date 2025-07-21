"""
CSV エクスポートユーティリティ

V-2タスク: CSV エクスポートユーティリティ
学習終了時に `runs/YYYYMMDD/metrics.csv` を出力する機能を提供します。

使用例:
    from eval.export_csv import export_metrics_to_csv, export_from_tensorboard
    
    # TensorBoardLoggerから直接エクスポート
    logger = TensorBoardLogger(...)
    export_metrics_to_csv(logger, output_path="runs/20250121/metrics.csv")
    
    # TensorBoardログファイルからエクスポート
    export_from_tensorboard("runs/experiment1", "runs/20250121/metrics.csv")
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# TensorBoardEventファイル読み込み用（オプション）
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


def export_metrics_to_csv(
    logger: 'TensorBoardLogger', 
    output_path: Optional[str] = None,
    include_timestamp: bool = True
) -> str:
    """
    TensorBoardLoggerからメトリクスをCSVにエクスポート
    
    Args:
        logger: TensorBoardLoggerインスタンス
        output_path: 出力パス（省略時は自動生成）
        include_timestamp: タイムスタンプを含めるかどうか
        
    Returns:
        作成されたCSVファイルのパス
    """
    # 出力パスの自動生成
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "metrics.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # メトリクス履歴取得
    metrics_history = logger.get_metrics_history()
    
    if not metrics_history:
        raise ValueError("メトリクス履歴が空です。ログを記録してからエクスポートしてください。")
    
    # CSV形式に変換
    csv_data = _convert_metrics_to_csv_format(metrics_history, include_timestamp)
    
    # CSV書き込み
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # ヘッダー行
        writer.writerow(csv_data['headers'])
        
        # データ行
        for row in csv_data['rows']:
            writer.writerow(row)
    
    print(f"メトリクスをCSVファイルにエクスポートしました: {output_path}")
    return str(output_path)


def export_from_tensorboard(
    tensorboard_log_dir: str,
    output_path: Optional[str] = None,
    include_timestamp: bool = True
) -> str:
    """
    TensorBoardログファイルから直接CSVにエクスポート
    
    Args:
        tensorboard_log_dir: TensorBoardログディレクトリ
        output_path: 出力パス（省略時は自動生成）
        include_timestamp: タイムスタンプを含めるかどうか
        
    Returns:
        作成されたCSVファイルのパス
    """
    if not HAS_TENSORBOARD:
        raise ImportError("TensorBoardがインストールされていません。pip install tensorboard でインストールしてください。")
    
    # 出力パスの自動生成
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("runs") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "metrics.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # TensorBoardイベントファイル読み込み
    event_accumulator = EventAccumulator(tensorboard_log_dir)
    event_accumulator.Reload()
    
    # スカラーメトリクス取得
    scalar_keys = event_accumulator.Tags()['scalars']
    metrics_data = {}
    
    for key in scalar_keys:
        scalar_events = event_accumulator.Scalars(key)
        metrics_data[key] = [(event.step, event.value) for event in scalar_events]
    
    if not metrics_data:
        raise ValueError(f"TensorBoardログディレクトリにスカラーデータが見つかりません: {tensorboard_log_dir}")
    
    # CSV形式に変換
    csv_data = _convert_metrics_to_csv_format(metrics_data, include_timestamp)
    
    # CSV書き込み
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # ヘッダー行
        writer.writerow(csv_data['headers'])
        
        # データ行
        for row in csv_data['rows']:
            writer.writerow(row)
    
    print(f"TensorBoardログからメトリクスをCSVファイルにエクスポートしました: {output_path}")
    return str(output_path)


def _convert_metrics_to_csv_format(
    metrics_data: Dict[str, List[Tuple[int, float]]], 
    include_timestamp: bool = True
) -> Dict[str, Any]:
    """
    メトリクス辞書をCSV形式に変換（内部関数）
    
    Args:
        metrics_data: メトリクス辞書 {metric_name: [(step, value), ...]}
        include_timestamp: タイムスタンプを含めるかどうか
        
    Returns:
        CSV形式辞書 {'headers': [...], 'rows': [...]}
    """
    # 全てのステップ番号を収集
    all_steps = set()
    for metric_values in metrics_data.values():
        all_steps.update(step for step, _ in metric_values)
    
    sorted_steps = sorted(all_steps)
    
    # ヘッダー行構築
    headers = ['episode']
    if include_timestamp:
        headers.append('timestamp')
    
    # メトリクス名をカテゴリ別にソート
    metric_names = sorted(metrics_data.keys())
    headers.extend(metric_names)
    
    # データ行構築
    rows = []
    for step in sorted_steps:
        row = [step]
        
        if include_timestamp:
            # ISO形式のタイムスタンプ（データ作成時点）
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row.append(timestamp)
        
        # 各メトリクスの値を取得（該当ステップに値がない場合はNaN）
        for metric_name in metric_names:
            metric_values = dict(metrics_data[metric_name])
            value = metric_values.get(step, np.nan)
            row.append(value)
        
        rows.append(row)
    
    return {
        'headers': headers,
        'rows': rows
    }


def create_experiment_summary(
    csv_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    CSVファイルから実験サマリーを作成
    
    Args:
        csv_path: CSVファイルのパス
        output_path: サマリー出力パス（省略時は同じディレクトリに作成）
        
    Returns:
        作成されたサマリーファイルのパス
    """
    csv_path = Path(csv_path)
    
    if output_path is None:
        output_path = csv_path.parent / "experiment_summary.txt"
    else:
        output_path = Path(output_path)
    
    # CSV読み込み
    metrics_data = {}
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, value in row.items():
                if key not in ['episode', 'timestamp']:
                    if key not in metrics_data:
                        metrics_data[key] = []
                    try:
                        metrics_data[key].append(float(value))
                    except (ValueError, TypeError):
                        pass  # NaNや空値をスキップ
    
    # サマリー統計計算
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== 実験サマリー ===\n")
        f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CSVファイル: {csv_path.name}\n\n")
        
        for metric_name, values in sorted(metrics_data.items()):
            if not values:
                continue
                
            f.write(f"【{metric_name}】\n")
            f.write(f"  平均: {np.mean(values):.4f}\n")
            f.write(f"  最小: {np.min(values):.4f}\n")
            f.write(f"  最大: {np.max(values):.4f}\n")
            f.write(f"  標準偏差: {np.std(values):.4f}\n")
            f.write(f"  データ数: {len(values)}\n")
            f.write(f"  最終値: {values[-1]:.4f}\n")
            f.write("\n")
    
    print(f"実験サマリーを作成しました: {output_path}")
    return str(output_path)


def batch_export_experiments(
    runs_dir: str = "runs",
    output_base_dir: Optional[str] = None
) -> List[str]:
    """
    複数の実験ディレクトリからバッチでCSVエクスポート
    
    Args:
        runs_dir: 実験ディレクトリのベースパス
        output_base_dir: 出力ベースディレクトリ（省略時はruns_dirと同じ）
        
    Returns:
        作成されたCSVファイルパスのリスト
    """
    if not HAS_TENSORBOARD:
        raise ImportError("TensorBoardがインストールされていません。pip install tensorboard でインストールしてください。")
    
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        raise ValueError(f"実験ディレクトリが見つかりません: {runs_dir}")
    
    if output_base_dir is None:
        output_base_dir = runs_dir
    output_base_path = Path(output_base_dir)
    
    csv_files = []
    
    # 各実験ディレクトリをスキャン
    for experiment_dir in runs_path.iterdir():
        if not experiment_dir.is_dir():
            continue
            
        # TensorBoardイベントファイルがあるかチェック
        event_files = list(experiment_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue
            
        try:
            # CSVエクスポート
            output_path = output_base_path / f"{experiment_dir.name}_metrics.csv"
            csv_path = export_from_tensorboard(
                str(experiment_dir),
                str(output_path),
                include_timestamp=True
            )
            csv_files.append(csv_path)
            
            # サマリー作成
            create_experiment_summary(csv_path)
            
        except Exception as e:
            print(f"実験 {experiment_dir.name} のエクスポートに失敗: {e}")
            continue
    
    print(f"バッチエクスポートが完了しました。{len(csv_files)}個のCSVファイルを作成しました。")
    return csv_files


__all__ = [
    'export_metrics_to_csv', 
    'export_from_tensorboard', 
    'create_experiment_summary',
    'batch_export_experiments'
]