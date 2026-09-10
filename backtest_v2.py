#!/usr/bin/env python3
"""Strict walk-forward / out-of-sample backtest for Loo-bot.

The target draw is never included in the feature/training history.
Compares Frequency, Poisson, ML, Frequency+Poisson, Full Loo-bot,
and an analytical random baseline.
"""

import json
import os
from math import comb
from datetime import datetime

import pandas as pd

from analyzer import (
    frequency_analysis,
    poisson_fairness_test,
    build_ml_features,
)
from config import GAMES, MIN_DRAWS_FOR_ML, MAX_DRAWS_WINDOW

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


OUT_DIR = "."


def norm01(values):
    s = pd.Series(values, dtype=float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def ml_scores(history, num_cols, number_range):
    if XGBClassifier is None or len(history) < MIN_DRAWS_FOR_ML:
        return pd.Series(0.5, index=range(number_range[0], number_range[1] + 1))

    features = build_ml_features(
        history,
        num_cols,
        number_range,
        window_sizes=(10, 30, 100),
    )
    if features.empty or "label" not in features.columns:
        return pd.Series(0.5, index=range(number_range[0], number_range[1] + 1))

    feature_cols = [
        c for c in features.columns
        if c not in {"label", "draw_index", "number"}
    ]
    train = features.dropna(subset=["label"])
    next_draw = features[features["label"].isna()].copy()
    if train.empty or next_draw.empty:
        return pd.Series(0.5, index=range(number_range[0], number_range[1] + 1))

    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train[feature_cols], train["label"].astype(int))
    probs = model.predict_proba(next_draw[feature_cols])[:, 1]
    return pd.Series(probs, index=next_draw["number"].astype(int))


def pick_numbers(score, pick, ordered=False):
    ranked = score.sort_values(ascending=False)
    chosen = list(ranked.head(pick).index)
    if not ordered:
        chosen.sort()
    return chosen


def random_expected(number_range, pick, ordered):
    n = number_range[1] - number_range[0] + 1
    overlap = (pick * pick) / n
    exact_p = (1 / (n ** pick)) if ordered else (1 / comb(n, pick))
    return overlap, exact_p


def actual_numbers(row, num_cols):
    return [int(row[c]) for c in num_cols if pd.notna(row[c])]


def score_target(history, game):
    num_cols = [f"n{i}" for i in range(1, game["pick"] + 1)]
    number_range = tuple(game["range"])

    counts = frequency_analysis(history, num_cols, number_range)
    z = poisson_fairness_test(
        counts,
        len(history),
        game["pick"],
        number_range,
    )["z_scores"]

    freq = pd.Series(counts, dtype=float)
    z = pd.Series(z, dtype=float)
    ml = ml_scores(history, num_cols, number_range)

    freq_n = norm01(freq)
    z_n = norm01(z)
    ml_n = norm01(ml)

    scores = {
        "Frequency": freq_n,
        "Poisson": z_n,
        "ML": ml_n,
        "Frequency+Poisson": 0.5 * freq_n + 0.5 * z_n,
        "Full Loo-bot": 0.25 * freq_n + 0.25 * z_n + 0.50 * ml_n,
    }
    return {name: pick_numbers(s, game["pick"], game["ordered"]) for name, s in scores.items()}


def overlap(pred, actual):
    return len(set(pred) & set(actual)) if pred and actual else 0


def exact_hit(pred, actual, ordered):
    if ordered:
        return pred == actual
    return sorted(pred) == sorted(actual)


def prepare_rows(csv_path, game):
    num_cols = [f"n{i}" for i in range(1, game["pick"] + 1)]
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Multi-draw games are stored in draw order per date; keep that order.
    if game.get("multi_draw_per_day"):
        slots = ["2:00 PM", "5:00 PM", "9:00 PM"]
        df["slot"] = df.groupby(df["date"].dt.date).cumcount().map(
            lambda i: slots[i] if i < len(slots) else f"slot-{i+1}"
        )
    else:
        df["slot"] = ""
    return df, num_cols


def run_game(game_name, game):
    csv_path = game["file"]
    if not os.path.exists(csv_path):
        return []

    df, num_cols = prepare_rows(csv_path, game)
    rows = []
    min_history = max(MIN_DRAWS_FOR_ML, 10)
    if len(df) <= min_history:
        return rows

    for target_i in range(min_history, len(df)):
        history = df.iloc[max(0, target_i - MAX_DRAWS_WINDOW):target_i].copy()
        target = df.iloc[target_i]
        actual = actual_numbers(target, num_cols)
        predictions = score_target(history, game)
        exp_overlap, exact_p = random_expected(game["range"], game["pick"], game["ordered"])

        for method, pred in predictions.items():
            rows.append({
                "game": game_name,
                "date": target["date"].strftime("%Y-%m-%d"),
                "slot": target["slot"],
                "method": method,
                "prediction": "-".join(f"{x:02d}" for x in pred),
                "actual": "-".join(f"{x:02d}" for x in actual),
                "matches": overlap(pred, actual),
                "exact_hit": int(exact_hit(pred, actual, game["ordered"])),
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
                random_expected_matches=("random_expected_matches", "mean"),
                random_exact_probability=("random_exact_probability", "mean"),
            )
        )
        summary["match_ratio_vs_random"] = (
            summary["total_matches"] /
            (summary["predictions"] * summary["random_expected_matches"])
        )
        summary["exact_hit_rate"] = summary["exact_hits"] / summary["predictions"]

    summary.to_csv(os.path.join(OUT_DIR, "backtest_v2_summary.csv"), index=False)
    with open(os.path.join(OUT_DIR, "backtest_v2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2, default=str)

    print(f"Backtest complete: {len(results)} method-results")
    print("Strict walk-forward: target draw excluded from all training/features.")


if __name__ == "__main__":
    main()
