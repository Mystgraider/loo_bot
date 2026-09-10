#!/usr/bin/env python3
"""Strict walk-forward backtest for production scoring and random baseline."""

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
    """Score the next draw using only the supplied historical rows."""
    pick = game["pick"]
    number_range = tuple(game["range"])
    num_cols = [f"n{i}" for i in range(1, pick + 1)]

    counts = frequency_analysis(history, num_cols, number_range)
    poisson = poisson_fairness_test(counts, len(history), pick, number_range)
    training_df, next_draw_df = build_ml_features(history, num_cols, number_range)
    ml, ml_used = train_and_score(
        training_df, next_draw_df, number_range, MIN_DRAWS_FOR_ML
    )

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
    """Convert a score vector into a prediction with production semantics."""
    ranked = score.sort_values(ascending=False)

    if game.get("type") == "combo" and not game.get("order_matters", False):
        return sorted(int(x) for x in ranked.head(pick).index)

    if game.get("allow_repetition", False):
        values = np.asarray(ranked.index, dtype=int)
        weights = np.clip(ranked.to_numpy(dtype=float), 0, None)
        if weights.sum() <= 0 or not np.isfinite(weights).all():
            weights = np.ones(len(values), dtype=float)
        rng = np.random.default_rng(seed)
        return [
            int(x)
            for x in rng.choice(
                values, size=pick, replace=True, p=weights / weights.sum()
            )
        ]

    return [int(x) for x in ranked.head(pick).index]


def random_prediction(number_range, pick, game, seed):
    """Generate a genuine random baseline using the game's draw semantics."""
    lo, hi = number_range
    values = np.arange(lo, hi + 1)
    rng = np.random.default_rng(seed)

    if game.get("allow_repetition", False):
        return [int(x) for x in rng.choice(values, size=pick, replace=True)]

    sampled = rng.choice(values, size=pick, replace=False)
    sampled = [int(x) for x in sampled]
    if not game.get("order_matters", False):
        sampled.sort()
    return sampled


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
    """Return theoretical random overlap and exact-hit probability.

    For repeated-digit games, the simple p^2/n overlap expression is only an
    approximation because multiset overlap caps duplicate matches. The actual
    random baseline is therefore also generated during the backtest and is the
    primary comparison for digit games.
    """
    n = number_range[1] - number_range[0] + 1
    expected_overlap = (pick * pick) / n

    if game.get("order_matters", False):
        if game.get("allow_repetition", False):
            denominator = n ** pick
        else:
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
            # The target row is deliberately excluded from every feature and
            # model fit. The history is capped exactly like production.
            history = series.iloc[max(0, target_i - MAX_DRAWS_WINDOW):target_i].copy()
            target = series.iloc[target_i]
            actual = actual_numbers(target, num_cols)
            scores, ml_used = score_target(history, game)
            date_str = target["date"].strftime("%Y-%m-%d")

            methods = {
                method: pick_numbers(
                    score, game["pick"], game, _seed(method, date_str, slot)
                )
                for method, score in scores.items()
            }
            methods["Random baseline"] = random_prediction(
                game["range"],
                game["pick"],
                game,
                _seed("Random baseline", date_str, slot),
            )

            exp_overlap, exact_p = random_expected(game["range"], game["pick"], game)
            for method, pred in methods.items():
                rows.append(
                    {
                        "game": game_name,
                        "date": date_str,
                        "slot": slot,
                        "method": method,
                        "prediction": "-".join(f"{x:02d}" for x in pred),
                        "actual": "-".join(f"{x:02d}" for x in actual),
                        "matches": overlap(
                            pred, actual, game.get("allow_repetition", False)
                        ),
                        "exact_hit": int(
                            exact_hit(pred, actual, game.get("order_matters", False))
                        ),
                        "ml_used": int(ml_used),
                        "random_expected_matches": exp_overlap,
                        "random_exact_probability": exact_p,
                    }
                )
    return rows


def wilson_interval(successes, trials, z=1.96):
    """Approximate 95% Wilson interval for an exact-hit rate."""
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def build_summary(results):
    if results.empty:
        return pd.DataFrame()

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
    summary["match_ratio_vs_theoretical_random"] = summary["total_matches"] / (
        summary["predictions"] * summary["random_expected_matches"]
    )
    summary["exact_hit_rate"] = summary["exact_hits"] / summary["predictions"]

    intervals = summary.apply(
        lambda r: wilson_interval(int(r["exact_hits"]), int(r["predictions"])),
        axis=1,
    )
    summary["exact_hit_rate_ci95_low"] = [x[0] for x in intervals]
    summary["exact_hit_rate_ci95_high"] = [x[1] for x in intervals]

    random_rates = (
        summary[summary["method"] == "Random baseline"]
        .set_index("game")["avg_matches"]
        .to_dict()
    )
    summary["avg_matches_vs_actual_random"] = summary.apply(
        lambda r: (
            r["avg_matches"] / random_rates[r["game"]]
            if r["game"] in random_rates and random_rates[r["game"]] > 0
            else np.nan
        ),
        axis=1,
    )
    return summary


def main():
    all_rows = []
    for game_name, game in GAMES.items():
        all_rows.extend(run_game(game_name, game))

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(OUT_DIR, "backtest_v2_results.csv"), index=False)
    summary = build_summary(results)
    summary.to_csv(os.path.join(OUT_DIR, "backtest_v2_summary.csv"), index=False)

    metadata = {
        "backtest": "strict_walk_forward_v3",
        "target_excluded_from_history": True,
        "max_history_draws": MAX_DRAWS_WINDOW,
        "min_history_draws": max(MIN_DRAWS_FOR_ML, 10),
        "methods": sorted(results["method"].unique().tolist()) if not results.empty else [],
        "note": "Random baseline is generated per target; theoretical overlap is approximate for repeated-digit multiset overlap.",
    }
    payload = {
        "metadata": metadata,
        "summary": summary.to_dict(orient="records"),
    }
    with open(os.path.join(OUT_DIR, "backtest_v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"Backtest complete: {len(results)} method-results")
    print("Strict walk-forward: target draw excluded from all feature/training history.")
    print("Random baseline: generated independently for every target draw.")


if __name__ == "__main__":
    main()
