import argparse
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


def parse_log(path: str) -> Tuple[List[float], List[float]]:
    """Parse log and return cumulative average reward and win rate."""
    encodings = ["utf-8", "utf-8-sig", "utf-16"]
    text = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode log file")

    pattern = re.compile(r"(?:Episode|Battle)\s+(\d+)\s+reward(?:=|\s)(-?\d+(?:\.\d+)?)\b.*?(?:win=(True|False))?", re.IGNORECASE)
    rewards: List[float] = []
    wins: List[int] = []
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            reward = float(m.group(2))
            win = m.group(3)
            rewards.append(reward)
            wins.append(1 if win == "True" else 0)

    cumulative_avg: List[float] = []
    cumulative_win: List[float] = []
    total_reward = 0.0
    wins_so_far = 0
    for i, (r, w) in enumerate(zip(rewards, wins), start=1):
        total_reward += r
        wins_so_far += w
        cumulative_avg.append(total_reward / i)
        cumulative_win.append(wins_so_far / i)

    return cumulative_avg, cumulative_win


def plot_comparison(
    data_a: Tuple[List[float], List[float]],
    data_b: Tuple[List[float], List[float]],
    labels: Tuple[str, str],
    output: str,
) -> None:
    """Plot average reward and win rate comparison."""
    avg_a, win_a = data_a
    avg_b, win_b = data_b
    episodes = range(1, max(len(avg_a), len(avg_b)) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(range(1, len(avg_a) + 1), avg_a, marker="o", label=labels[0])
    ax1.plot(range(1, len(avg_b) + 1), avg_b, marker="o", label=labels[1])
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Average Reward per Episode")
    ax1.legend()

    ax2.plot(range(1, len(win_a) + 1), win_a, marker="o", label=labels[0])
    ax2.plot(range(1, len(win_b) + 1), win_b, marker="o", label=labels[1])
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate")
    ax2.set_ylim(0, 1)
    ax2.set_title("Cumulative Win Rate")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output)
    print(f"Saved plot to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare RL logs")
    parser.add_argument("log_a", type=str, help="path to first log file")
    parser.add_argument("log_b", type=str, help="path to second log file")
    parser.add_argument("--out", type=str, default="compare.png", help="output image file path")
    parser.add_argument("--labels", nargs=2, metavar=("A", "B"), default=["REINFORCE", "PPO"], help="labels for the two logs")
    args = parser.parse_args()

    data_a = parse_log(args.log_a)
    data_b = parse_log(args.log_b)
    plot_comparison(data_a, data_b, tuple(args.labels), args.out)
