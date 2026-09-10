#!/usr/bin/env python3
"""Strict walk-forward backtest that shares the production analyzer API."""

import hashlib
import json
import os
from collections import Counter
from math import comb

import numpy as np
import pandas as pd

from analyzer import (
    frequency_analysis,
    poisson_fairness_test,
    build_ml_features,
    train_and_score,
    combined_score,
)
from config import GAMES, MIN_DRAWS_FOR_ML, MAX_DRAWS_WINDOW, SLOT_LABELS
from schema import validate_dataframe

OUT_DIR = "."


def norm01(values):
    s = pd.Series(values, dtype=float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def score_target(history, game):
    pick = game["pick"]
    number_range = tuple(game["range"])
    num_cols = [f"n{i}" for i in range(1, pick + 1)]

    counts = frequency_analysis(history, num_cols, number_range)
    poisson = poisson_fairness_test(counts, len(history), pick, number_range)
    training_df, next_draw_df = build_ml_features(history, num_cols, number_range)
    ml, ml_used = train_and_score(training_df, next_draw_df, number_range, MIN_DRAWS_FOR_ML)

    freq_n = norm01(counts)
    z_n = norm01(poisson["z_scores"])
    ml_n = norm01(ml)
    return {
        "Frequency": freq_n,
        "Poisson": z_n,
        "ML": ml_n,
        "Frequency+Poisson": 0.5 * freq_n + 0.5 * z_n,
        "Full Loo-bot": 0.25 * freq_n + 0.25 * z_n + 0.50 * ml_n,
    }, ml_used


def _seed(method, date_str, slot):
    raw = f"{method}|{date_str}|{slot}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def pick_numbers(score, pick, game, seed):
    ranked = score.sort_values(ascending=False)
    if game.get("type") == "combo" and not game.get("order_matters", False):
        return sorted(int(x) for x in ranked.head(pick).index)

    # Ordered digit games may repeat digits. Weighted sampling with replacement
    # mirrors the production selection semantics without pretending it is a true
    # probability forecast.
    if game.get("allow_repetition", False):
        values = np.asarray(ranked.index, dtype=int)
        weights = np.clip(ranked.to_numpy(dtype=float), 0, None)
        if weights.sum() <= 0 or not np.isfinite(weights).all():
            weights = np.ones(len(values), dtype=float)
        rng = np.random.default_rng(seed)
        return [int(x) for x in rng.choice(values, size=pick, replace=True, p=weights / weights.sum())]

    return [int(x) for x in ranked.head(pick).index]


def actual_numbers(row, num_cols):
    return [int(row[c]) for c in num_cols if pd.notna(row[c])]


def overlap(pred, actual, allow_repetition=False):
    if not pred or not actual:
        return 0
    if not allow_repetition:
        return len(set(pred) & set(actual))
    return sum((Counter(pred) & Counter(actual)).values())


def exact_hit(pred, actual, order_matters):
    return pred == actual if order_matters else sorted(pred) == sorted(actual)


def random_expected(number_range, pick, game):
    n = number_range[1] - number_range[0] + 1
    expected_overlap = (pick * pick) / n
    if game.get("order_matters", False):
        exact_p = 1 / (n ** pick) if game.get("allow_repetition", False) else 1 / (n * (n - 1) * max(1, np.prod(range(n - pick + 1, n + 1))))
        # For the configured ordered non-repeating case (EZ2), the exact
        # probability is 1 / P(n,pick).
        if not game.get("allow_repetition", False):
            denominator = 1
            for k in range(pick):
                denominator *= n - k
            exact_p = 1 / denominator
    else:
        exact_p = 1 / comb(n, pick)
    return expected_overlap, exact_p


def prepare_rows(csv_path, game):
    pick = game["pick"]
    multi = game.get("multi_draw_per_day", False)
    df = pd.read_csv(csv_path)
    df = validate_dataframe(df, pick, tuple(game["range"]), multi_draw=multi)
    return df, [f"n{i}" for i in range(1, pick + 1)]


def iter_series(df, game):
    if not game.get("multi_draw_per_day", False):
        yield "", df.reset_index(drop=True)
        return
    for slot in SLOT_LABELS:
        yield slot, df[df["slot"] == slot].drop(columns=["slot"]).reset_index(drop=True)


def run_game(game_name, game):
    csv_path = game["csv"]
    if not os.path.exists(csv_path):
        return []

    df, num_cols = prepare_rows(csv_path, game)
    rows = []
    min_history = max(MIN_DRAWS_FOR_ML, 10)

    for slot, series in iter_series(df, game):
        if len(series) <= min_history:
            continue
        for target_i in range(min_history, len(series)):
            history = series.iloc[max(0, target_i - MAX_DRAWS_WINDOW):target_i].copy()
            target = series.iloc[target_i]
            actual = actual_numbers(target, num_cols)
            scores, ml_used = score_target(history, game)
            date_str = target["date"].strftime("%Y-%m-%d")
            for method, score in scores.items():
                pred = pick_numbers(score, game["pick"], game, _seed(method, date_str, slot))
                exp_overlap, exact_p = random_expected(game["range"], game["pick"], game)
                rows.append({
                    "game": game_name,
                    "date": date_str,
                    "slot": slot,
                    "method": method,
                    "prediction": "-".join(f"{x:02d}" for x in pred),
                    "actual": "-".join(f"{x:02d}" for x in actual),
                    "matches": overlap(pred, actual, game.get("allow_repetition", False)),
                    "exact_hit": int(exact_hit(pred, actual, game.get("order_matters", False))),
                    "ml_used": int(ml_used),
                    "random_expected_matches": exp_overlap,
                    "random_exact_probability": exact_p,
                })
    return rows


def main():
    all_rows = []
    for game_name, game in GAMES.items():
        all_rows.extend(run_game(game_name, game))

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(OUT_DIR, "backtest_v2_results.csv"), index=False)
    if results.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            results.groupby(["game", "method"], as_index=False)
            .agg(
                predictions=("matches", "size"),
                total_matches=("matches", "sum"),
                avg_matches=("matches", "mean"),
                exact_hits=("exact_hit", "sum"),
                ml_used_rate=("ml_used", "mean"),
                random_expected_matches=("random_expected_matches", "mean"),
                random_exact_probability=("random_exact_probability", "mean"),
            )
        )
        summary["match_ratio_vs_random"] = summary["total_matches"] / (
            summary["predictions"] * summary["random_expected_matches"]
        )
        summary["exact_hit_rate"] = summary["exact_hits"] / summary["predictions"]

    summary.to_csv(os.path.join(OUT_DIR, "backtest_v2_summary.csv"), index=False)
    with open(os.path.join(OUT_DIR, "backtest_v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2, default=str)

    print(f"Backtest complete: {len(results)} method-results")
    print("Strict walk-forward: target draw excluded from all feature/training history.")


if __name__ == "__main__":
    main()
