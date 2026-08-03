#!/usr/bin/env python3
"""
predict.py

Pangunahing script. Para sa bawat game sa config.py:
  1. Load-in ang historical draws (huling MAX_DRAWS_WINDOW draws lang)
  2. Frequency analysis
  3. Poisson fairness test
  4. ML scoring (kung sapat ang data)
  5. Bumuo ng "recommended" na combination base sa combined score

Output: results.json (para magamit ng ibang script, e.g. telegram_notify.py)
        at print sa console/log ng GitHub Actions.
"""

import json
import os
import random
from datetime import datetime, timezone

import numpy as np

from config import GAMES, MIN_DRAWS_FOR_ML, MAX_DRAWS_WINDOW
from analyzer import (
    load_draws,
    frequency_analysis,
    poisson_fairness_test,
    build_ml_features,
    train_and_score,
    combined_score,
)

DISCLAIMER = (
    "⚠️ Disclaimer: Statistical exploration lang ito (frequency + Poisson + ML). "
    "Independent random event ang bawat lotto draw -- hindi ito totoong predictive "
    "at hindi ka dapat mag-bet base lang dito. Para sa libangan/research purposes lang."
)


def pick_combo_numbers(scores, pick, number_range):
    """Pumili ng top-N na numero na may pinakamataas na combined score, unique."""
    top = list(scores.index[:pick])
    return sorted(int(n) for n in top)


def pick_digit_numbers(scores, pick, number_range):
    """Para sa digit games: puwedeng magulit ang digit sa bawat posisyon,
    kaya pumili tayo ng top-`pick` na may pinakamataas na score, kahit magkapareho."""
    # kunin ang top scores, puwedeng ulitin ang parehong digit sa magkaibang posisyon
    ranked = scores.sort_values(ascending=False)
    chosen = []
    idx = 0
    ranked_list = list(ranked.index)
    while len(chosen) < pick:
        chosen.append(int(ranked_list[idx % len(ranked_list)]))
        idx += 1
    return chosen


def analyze_game(game_name, game_cfg):
    csv_path = game_cfg["csv"]
    pick = game_cfg["pick"]
    number_range = game_cfg["range"]

    if not os.path.exists(csv_path):
        return {
            "game": game_name,
            "error": f"Walang nahanap na data file: {csv_path}. "
                     f"Maglagay ng CSV (columns: date,n1,...,n{pick}) para masuri ito.",
        }

    df, num_cols = load_draws(csv_path, pick)
    df = df.tail(MAX_DRAWS_WINDOW).reset_index(drop=True)
    n_draws = len(df)

    if n_draws < 10:
        return {
            "game": game_name,
            "error": f"Kulang ang historical data ({n_draws} draws lang). "
                     f"Kailangan ng mas maraming rows sa {csv_path}.",
        }

    freq_counts = frequency_analysis(df, num_cols, number_range)
    poisson_result = poisson_fairness_test(freq_counts, n_draws, pick, number_range)

    training_df, next_draw_df = build_ml_features(df, num_cols, number_range)
    ml_probs, ml_used = train_and_score(training_df, next_draw_df, number_range, MIN_DRAWS_FOR_ML)

    scores = combined_score(freq_counts, poisson_result, ml_probs, ml_used)

    if game_cfg["type"] == "combo":
        recommendation = pick_combo_numbers(scores, pick, number_range)
    else:
        recommendation = pick_digit_numbers(scores, pick, number_range)

    return {
        "game": game_name,
        "n_draws_analyzed": n_draws,
        "ml_used": ml_used,
        "poisson_p_value": round(poisson_result["p_value"], 4),
        "significantly_biased": poisson_result["is_significantly_biased"],
        "recommendation": recommendation,
        "top_10_by_score": [int(x) for x in scores.index[:10]],
    }


def main():
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
        "games": [],
    }

    for game_name, game_cfg in GAMES.items():
        try:
            result = analyze_game(game_name, game_cfg)
        except Exception as e:
            result = {"game": game_name, "error": str(e)}
        results["games"].append(result)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nNasave sa results.json")


if __name__ == "__main__":
    main()
