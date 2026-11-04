#!/usr/bin/env python3
"""Experimental Aklan STL Pares analysis.

CSV contract: date,slot,n1,n2,source
slot is exactly "2:00 PM" or "7:00 PM"; numbers are 1..40.
This module deliberately does not scrape an unverified third-party source.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

SLOTS = ("2:00 PM", "7:00 PM")
NUMBER_MIN, NUMBER_MAX = 1, 40
PAIR_SPACE = 40 * 40


def load_history(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    seen = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "slot", "n1", "n2"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"STL CSV must contain columns: {', '.join(sorted(required))}")
        for line, raw in enumerate(reader, start=2):
            date, slot = (raw.get("date") or "").strip(), (raw.get("slot") or "").strip()
            try:
                n1, n2 = int(raw["n1"]), int(raw["n2"])
            except (TypeError, ValueError):
                raise ValueError(f"Invalid pair on CSV line {line}") from None
            if slot not in SLOTS:
                raise ValueError(f"Unsupported slot {slot!r} on CSV line {line}")
            if not (NUMBER_MIN <= n1 <= NUMBER_MAX and NUMBER_MIN <= n2 <= NUMBER_MAX):
                raise ValueError(f"Numbers must be 01..40 on CSV line {line}")
            key = (date, slot)
            if not date or key in seen:
                raise ValueError(f"Missing date or duplicate draw {key} on CSV line {line}")
            seen.add(key)
            rows.append({"date": date, "slot": slot, "n1": n1, "n2": n2,
                         "source": (raw.get("source") or "").strip()})
    return sorted(rows, key=lambda x: (x["date"], SLOTS.index(x["slot"])))


def rank_pairs(history: list[dict], slot: str, top_n: int = 10) -> list[dict]:
    """Rank exact ordered pairs with shrinkage; no future rows are consulted."""
    rows = [r for r in history if r["slot"] == slot]
    if len(rows) < 20:
        return []
    first, second, pairs = Counter(), Counter(), Counter()
    for row in rows:
        first[row["n1"]] += 1
        second[row["n2"]] += 1
        pairs[(row["n1"], row["n2"])] += 1

    n = len(rows)
    ranked = []
    # Empirical-Bayes-style smoothing toward uniform pair probability.
    # The 20-draw prior strength prevents tiny samples from dominating.
    prior_strength = 20.0
    for a in range(1, 41):
        for b in range(1, 41):
            observed = pairs[(a, b)]
            pair_rate = (observed + prior_strength / PAIR_SPACE) / (n + prior_strength)
            pos1_rate = (first[a] + 1.0) / (n + 40.0)
            pos2_rate = (second[b] + 1.0) / (n + 40.0)
            # Pair co-occurrence carries most weight; marginal position rates
            # are weak tie-breakers, not claims of causal predictability.
            score = 0.80 * pair_rate + 0.10 * pos1_rate + 0.10 * pos2_rate
            ranked.append({"pair": [a, b], "score": score, "observed_pair_count": observed})
    ranked.sort(key=lambda x: (-x["score"], x["pair"][0], x["pair"][1]))
    return ranked[:max(1, min(int(top_n), PAIR_SPACE))]


def walk_forward_backtest(history: list[dict], slot: str, min_train: int = 30,
                          top_k: int = 10) -> dict:
    """Chronological expanding-window test; each target is scored from past rows only."""
    rows = [r for r in history if r["slot"] == slot]
    hits = 0
    tested = 0
    for index in range(min_train, len(rows)):
        ranked = rank_pairs(rows[:index], slot, top_k)
        if not ranked:
            continue
        target = (rows[index]["n1"], rows[index]["n2"])
        tested += 1
        if any(tuple(item["pair"]) == target for item in ranked):
            hits += 1
    random_expected = top_k / PAIR_SPACE
    return {
        "slot": slot, "history_draws": len(rows), "min_train": min_train,
        "top_k": top_k, "tested_draws": tested, "hits": hits,
        "hit_rate": round(hits / tested, 6) if tested else None,
        "random_baseline_hit_rate": random_expected,
        "random_expected_hits": round(tested * random_expected, 4),
        "beats_random_observed": bool(tested and hits / tested > random_expected),
        "status": "EVALUATED" if tested else "INSUFFICIENT_HISTORY",
        "warning": "Small samples are noisy; beating the theoretical baseline once is not proof of an edge."
    }


def build_report(history: list[dict]) -> dict:
    report = {"status": "RESEARCH_ONLY", "draws_loaded": len(history), "slots": {}}
    for slot in SLOTS:
        slot_rows = [r for r in history if r["slot"] == slot]
        report["slots"][slot] = {
            "draws": len(slot_rows),
            "backtest": walk_forward_backtest(history, slot),
            "candidates": [
                {"pair": item["pair"], "score": round(item["score"], 8),
                 "observed_pair_count": item["observed_pair_count"]}
                for item in rank_pairs(history, slot, 5)
            ],
            "prediction_status": "AVAILABLE_FOR_RESEARCH" if rank_pairs(history, slot, 5)
                                 else "INSUFFICIENT_VERIFIED_DATA"
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/stl_aklan_pares.csv")
    parser.add_argument("--output", default="stl_pares_results.json")
    args = parser.parse_args()
    history = load_history(args.csv)
    report = build_report(history)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
