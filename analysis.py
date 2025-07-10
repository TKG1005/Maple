
import os
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import sys

def analyze_tensorboard_logs(log_dirs, configs):
    results = []
    if len(log_dirs) != len(configs):
        print(f"Error: Mismatch between number of directories ({len(log_dirs)}) and configs ({len(configs)}).")
        num_to_process = min(len(log_dirs), len(configs))
        log_dirs = log_dirs[:num_to_process]
        configs = configs[:num_to_process]
        print(f"Processing the first {num_to_process} directories.")

    for log_dir, config in zip(log_dirs, configs):
        try:
            event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
            if not event_files:
                print(f"Warning: No event file found in {log_dir}")
                continue
            event_path = os.path.join(log_dir, event_files[0])
            
            ea = event_accumulator.EventAccumulator(event_path, size_guidance={event_accumulator.SCALARS: 0})
            ea.Reload()
            
            tag = 'time/episode'
            if tag not in ea.Tags()['scalars']:
                print(f"Warning: Tag '{tag}' not found in {log_dir}")
                print(f"Available tags: {ea.Tags()['scalars']}")
                continue

            durations = [s.value for s in ea.Scalars(tag)]
            if durations:
                avg_duration_ms = sum(durations) / len(durations)
                results.append({
                    'Run': os.path.basename(log_dir),
                    'Parallelism': config[0],
                    'Device': config[1],
                    'Avg Episode Time (ms)': avg_duration_ms
                })
        except Exception as e:
            print(f'Could not process {log_dir}: {e}', file=sys.stderr)
            
    if not results:
        print("No data could be extracted from the logs.")
        return

    df = pd.DataFrame(results)
    if not df.empty:
        df['Total Time per Batch (s)'] = df['Avg Episode Time (ms)'] * df['Parallelism'] / 1000
        df = df.sort_values(by='Total Time per Batch (s)')
    print('--- Analysis Results ---')
    print(df.to_string())

if __name__ == "__main__":
    base_dir = 'runs'
    if not os.path.isdir(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        sys.exit(1)

    run_dirs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    # 全ての実行を設定
    run_configs = [
        (10, 'CPU'),       # 1st
        (10, 'CUDA'),      # 2nd
        (50, 'CUDA'),      # 3rd
        (1, 'CPU'),        # 4th
        (20, 'CUDA'),      # 5th (new)
        (30, 'CUDA'),      # 6th (new)
        (5, 'CUDA')        # 7th (new)
    ]
    
    analyze_tensorboard_logs(run_dirs, run_configs)
