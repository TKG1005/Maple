#!/usr/bin/env python3
"""
Aggregate [METRIC] log lines produced by Maple to compute baseline statistics.

Usage:
  python scripts/metrics_aggregator.py path/to/logfile.log [more.log ...]
  cat path/to/logfile.log | python scripts/metrics_aggregator.py -

Outputs a human-readable summary per tag and numeric key.
"""
from __future__ import annotations

import sys
import re
import math
from collections import defaultdict
from statistics import mean

METRIC_PREFIX = "[METRIC]"

def parse_line(line: str) -> tuple[str | None, dict[str, float]]:
    if METRIC_PREFIX not in line:
        return None, {}
    try:
        # Extract after prefix
        parts = line.strip().split(METRIC_PREFIX, 1)[1].strip().split()
        kv = {}
        tag = None
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            if k == "tag":
                tag = v
                continue
            # Strip trailing punctuation
            v = v.rstrip(",)")
            # Try numeric
            try:
                if v.lower().endswith("ms") and v[:-2].isdigit():
                    val = float(v[:-2])
                else:
                    val = float(v)
                kv[k] = val
            except Exception:
                # Non-numeric; skip
                continue
        return tag, kv
    except Exception:
        return None, {}

def percentile(values, p):
    if not values:
        return math.nan
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[int(f)] * (c - k)
    d1 = s[int(c)] * (k - f)
    return d0 + d1

def main(paths):
    lines = []
    if not paths or paths == ["-"]:
        lines = sys.stdin.read().splitlines()
    else:
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    lines.extend(f.readlines())
            except Exception as e:
                print(f"warn: failed to read {p}: {e}")

    # tag -> key -> list of values
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for line in lines:
        tag, kv = parse_line(line)
        if not tag or not kv:
            continue
        for k, v in kv.items():
            data[tag][k].append(v)

    if not data:
        print("no metric lines found. ensure code with S0 is running and logs are captured.")
        return 1

    print("METRIC SUMMARY")
    for tag in sorted(data.keys()):
        print(f"\n== tag: {tag} ==")
        for k, vals in sorted(data[tag].items()):
            if not vals:
                continue
            try:
                cnt = len(vals)
                mn = min(vals)
                mx = max(vals)
                avg = mean(vals)
                p50 = percentile(vals, 50)
                p95 = percentile(vals, 95)
                print(f"- {k}: count={cnt} min={mn:.1f} avg={avg:.1f} p50={p50:.1f} p95={p95:.1f} max={mx:.1f}")
            except Exception:
                continue
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

