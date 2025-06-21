import json
import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_results(path: str) -> List[dict]:
    """Load battle results list from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Invalid log format: 'results' list missing")
    return results


def plot_metrics(results: List[dict], output: str) -> None:
    """Plot reward history and cumulative win rate."""
    rewards = [float(r.get("reward", 0)) for r in results]
    wins = [1 if r.get("winner") == "env0" else 0 for r in results]

    cumulative_win_rate = []
    wins_so_far = 0
    for i, w in enumerate(wins, start=1):
        wins_so_far += w
        cumulative_win_rate.append(wins_so_far / i)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(range(1, len(rewards) + 1), rewards, marker="o")
    ax1.set_ylabel("Reward")
    ax1.set_title("Reward per Battle")

    ax2.plot(range(1, len(cumulative_win_rate) + 1), cumulative_win_rate, marker="o")
    ax2.set_xlabel("Battle")
    ax2.set_ylabel("Win Rate")
    ax2.set_ylim(0, 1)
    ax2.set_title("Cumulative Win Rate")

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RL evaluation results")
    parser.add_argument("log", type=str, help="path to JSON log file")
    parser.add_argument(
        "--out", type=str, default=None, help="output image file path (.png)"
    )
    args = parser.parse_args()

    out_path = args.out or str(Path(args.log).with_suffix(".png"))
    plot_metrics(load_results(args.log), out_path)
