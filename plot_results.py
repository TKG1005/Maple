import json
import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_results(path: str) -> List[dict]:
    """Load battle results list from a log file containing JSON output."""

    encodings = ["utf-8", "utf-8-sig", "utf-16"]
    text: str | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:  # pragma: no cover - fallback
            continue
    if text is None:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode log file")

    start_idx = text.rfind("{")
    if start_idx == -1:
        raise ValueError("No JSON object found in log")

    data = json.loads(text[start_idx:])
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
