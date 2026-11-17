#!/usr/bin/env python3
"""Production prediction runner using the shared analyzer and canonical schema."""

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    from zoneinfo import ZoneInfo
    PH_TZ = ZoneInfo("Asia/Manila")
except Exception:
    PH_TZ = None

from config import GAMES, MIN_DRAWS_FOR_ML, MAX_DRAWS_WINDOW, SLOT_LABELS
from analyzer import load_draws, frequency_analysis, poisson_fairness_test, build_ml_features, train_and_score, combined_score
from coverage_tracker import log_combo, get_coverage_stats

DISCLAIMER = (
    "⚠️ Disclaimer: Statistical exploration lang ito (frequency + Poisson + ML). "
    "Independent random event ang bawat lotto draw -- hindi ito totoong predictive "
    "at hindi ka dapat mag-bet base lang dito. Para sa libangan/research purposes lang."
)


def get_ph_today():
    if PH_TZ is not None:
        return datetime.now(PH_TZ).date()
    return datetime.now(timezone.utc).date()


def next_draw_date(draw_days, today):
    if draw_days is None:
        return today + timedelta(days=1)
    for offset in range(1, 8):
        candidate = today + timedelta(days=offset)
        if candidate.strftime("%A") in draw_days:
            return candidate
    return None


def split_by_slot(df):
    return {label: df[df["slot"] == label].drop(columns=["slot"]).reset_index(drop=True) for label in SLOT_LABELS}


def pick_combo_numbers(scores, pick):
    return sorted(int(n) for n in scores.index[:pick])


def pick_ordered_numbers(df, num_cols, pick, number_range, window=120, smoothing=1.0):
    """Rank each draw position independently; preserve order and valid repeats.

    The previous implementation collapsed ordered draws into an unordered
    digit-presence model, then sampled from one aggregate score vector. That
    discarded position information. This selector uses only prior draws and
    independently ranks the historical values at each position.
    """
    lo, hi = number_range
    history = df.tail(window)
    recommendation = []
    position_top = []
    for col in num_cols[:pick]:
        counts = history[col].value_counts().reindex(range(lo, hi + 1), fill_value=0)
        # Additive smoothing prevents zero-probability positions; it does not
        # change the ranking when the same prior is applied to every value.
        smoothed = counts.astype(float) + float(smoothing)
        ranked = sorted(smoothed.index, key=lambda value: (-smoothed.loc[value], int(value)))
        recommendation.append(int(ranked[0]))
        position_top.append([int(value) for value in ranked[:3]])
    return recommendation, position_top


def run_analysis(df, num_cols, game_cfg):
    pick = game_cfg["pick"]
    number_range = game_cfg["range"]
    order_matters = game_cfg.get("order_matters", False)
    freq_counts = frequency_analysis(df, num_cols, number_range)
    poisson_result = poisson_fairness_test(freq_counts, len(df), pick, number_range)

    # XGBoost's current labels represent number presence in a draw, not
    # position-specific outcomes. It is therefore not valid for ordered games.
    if order_matters:
        ml_probs = freq_counts.astype(float).mul(0).add(0.5)
        ml_used = False
    else:
        training_df, next_draw_df = build_ml_features(df, num_cols, number_range)
        ml_probs, ml_used = train_and_score(training_df, next_draw_df, number_range, MIN_DRAWS_FOR_ML)

    scores = combined_score(freq_counts, poisson_result, ml_probs, ml_used)

    position_top = None
    if order_matters:
        recommendation, position_top = pick_ordered_numbers(
            df, num_cols, pick, number_range
        )
        recommendation_method = "position_specific_historical_frequency"
    elif game_cfg["type"] == "combo":
        recommendation = pick_combo_numbers(scores, pick)
        recommendation_method = "frequency_plus_validated_model" if ml_used else "historical_frequency_baseline"
    else:
        recommendation = pick_combo_numbers(scores, pick)
        recommendation_method = "historical_frequency_baseline"

    return {
        "n_draws_analyzed": len(df),
        "ml_used": ml_used,
        "ordered": order_matters,
        "allow_repetition": game_cfg.get("allow_repetition", False),
        "poisson_p_value": round(poisson_result["p_value"], 4),
        "significantly_biased": poisson_result["is_significantly_biased"],
        "recommendation": recommendation,
        "top_10_by_score": [int(x) for x in scores.index[:10]],
    }


def analyze_game(game_name, game_cfg, today):
    csv_path = game_cfg["csv"]
    pick = game_cfg["pick"]
    is_multi = game_cfg.get("multi_draw_per_day", False)
    if not os.path.exists(csv_path):
        return [{"game": game_name, "error": f"Missing data file: {csv_path}"}]

    df, num_cols = load_draws(csv_path, pick, game_cfg["range"], multi_draw=is_multi)
    if not is_multi:
        df = df.tail(MAX_DRAWS_WINDOW).reset_index(drop=True)
        if len(df) < 10:
            return [{"game": game_name, "error": f"Insufficient historical data: {len(df)} draws."}]
        target_date = next_draw_date(game_cfg.get("draw_days"), today)
        result = run_analysis(df, num_cols, game_cfg)
        result.update({
            "game": game_name,
            "target_draw_date": target_date.isoformat() if target_date else None,
            "target_draw_weekday": target_date.strftime("%A") if target_date else None,
            "is_daily_draw": game_cfg.get("draw_days") is None,
        })
        return [result]

    target_date = next_draw_date(None, today)
    out = []
    for slot_label, slot_df in split_by_slot(df).items():
        slot_df = slot_df.tail(MAX_DRAWS_WINDOW).reset_index(drop=True)
        game_label = f"{game_name} ({slot_label})"
        if len(slot_df) < 10:
            out.append({"game": game_label, "error": f"Insufficient historical data: {len(slot_df)} draws."})
            continue
        result = run_analysis(slot_df, num_cols, game_cfg)
        result.update({
            "game": game_label,
            "target_draw_date": target_date.isoformat(),
            "target_draw_weekday": target_date.strftime("%A"),
            "target_draw_time": slot_label,
            "is_daily_draw": True,
        })
        if game_name == "ez2":
            log_combo(result["recommendation"])
        out.append(result)
    return out


def main():
    today = get_ph_today()
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_at_ph_date": today.isoformat(),
        "disclaimer": DISCLAIMER,
        "games": [],
    }
    for game_name, game_cfg in GAMES.items():
        try:
            game_results = analyze_game(game_name, game_cfg, today)
        except Exception as exc:
            game_results = [{"game": game_name, "error": str(exc)}]
        results["games"].extend(game_results)
        for result in game_results:
            print(json.dumps(result, indent=2, ensure_ascii=False))

    results["ez2_coverage"] = get_coverage_stats()
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    errors = [r for r in results["games"] if "error" in r]
    if errors:
        print(f"Prediction aborted: {len(errors)} game/slot errors.", file=sys.stderr)
        return 1
    print("\nNasave sa results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
