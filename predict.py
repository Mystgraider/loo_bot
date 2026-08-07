#!/usr/bin/env python3
"""
predict.py

Pangunahing script. Para sa bawat game sa config.py:
  1. Load-in ang historical draws (huling MAX_DRAWS_WINDOW draws lang)
  2. Frequency analysis
  3. Poisson fairness test
  4. ML scoring (kung sapat ang data)
  5. Bumuo ng "recommended" na combination base sa combined score
  6. Kalkulahin kung ANONG PETSA (at, sa EZ2/Swertres, ANONG ORAS) ang
     susunod na scheduled draw, para malinaw kung para saang draw ang
     suggestion.

Para sa mga "multi_draw_per_day" games (EZ2, Swertres -- 3x/araw),
HIWALAY ang analysis per time slot (2PM, 5PM, 9PM) -- ibig sabihin
magkaiba ang suggestion depende sa oras, hindi iisa lang para sa
buong araw.

Output: results.json (para magamit ng ibang script, e.g. telegram_notify.py)
        at print sa console/log ng GitHub Actions.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    from zoneinfo import ZoneInfo
    PH_TZ = ZoneInfo("Asia/Manila")
except Exception:
    PH_TZ = None

from config import GAMES, MIN_DRAWS_FOR_ML, MAX_DRAWS_WINDOW, SLOT_LABELS
from analyzer import (
    load_draws,
    frequency_analysis,
    poisson_fairness_test,
    build_ml_features,
    train_and_score,
    combined_score,
)
from coverage_tracker import log_combo, get_coverage_stats

DISCLAIMER = (
    "⚠️ Disclaimer: Statistical exploration lang ito (frequency + Poisson + ML). "
    "Independent random event ang bawat lotto draw -- hindi ito totoong predictive "
    "at hindi ka dapat mag-bet base lang dito. Para sa libangan/research purposes lang."
)


def get_ph_today():
    """Kunin ang 'ngayon' na petsa sa Manila time (hindi UTC, dahil doon
    tumatakbo ang GitHub Actions runner)."""
    if PH_TZ is not None:
        return datetime.now(PH_TZ).date()
    return datetime.now(timezone.utc).date()


def next_draw_date(draw_days, today):
    """Ibinabalik ang PETSA ng SUSUNOD na scheduled draw (susunod na araw
    palagi, kasi ipinapalagay na tapos na ang draw ng "today")."""
    if draw_days is None:
        return today + timedelta(days=1)
    for offset in range(1, 8):
        candidate = today + timedelta(days=offset)
        if candidate.strftime("%A") in draw_days:
            return candidate
    return None


def split_by_slot(df, num_cols):
    """Hinahati ang isang multi-draw-per-day na dataframe (palaging 3 rows
    kada date, sa pagkakasunod na 2PM->5PM->9PM) sa 3 HIWALAY na dataframe,
    isa per time slot -- para hiwalay ang trend/history ng bawat oras."""
    df = df.copy()
    df["_slot_idx"] = df.groupby("date").cumcount()
    slots = {}
    for i, label in enumerate(SLOT_LABELS):
        sub = df[df["_slot_idx"] == i].drop(columns=["_slot_idx"]).reset_index(drop=True)
        slots[label] = sub
    return slots


def pick_combo_numbers(scores, pick, number_range):
    top = list(scores.index[:pick])
    return sorted(int(n) for n in top)


def pick_digit_numbers(scores, pick, number_range):
    ranked = scores.sort_values(ascending=False)
    chosen = []
    idx = 0
    ranked_list = list(ranked.index)
    while len(chosen) < pick:
        chosen.append(int(ranked_list[idx % len(ranked_list)]))
        idx += 1
    return chosen


def run_analysis(df, num_cols, game_cfg):
    """Ang core na analysis (frequency + Poisson + ML + scoring) gamit ang
    kahit anong dataframe ng draws -- reusable para sa buong-araw na series
    (combo games) o per-slot na subset (EZ2/Swertres)."""
    pick = game_cfg["pick"]
    number_range = game_cfg["range"]
    ordered = game_cfg.get("ordered", False)
    n_draws = len(df)

    freq_counts = frequency_analysis(df, num_cols, number_range)
    poisson_result = poisson_fairness_test(freq_counts, n_draws, pick, number_range)

    training_df, next_draw_df = build_ml_features(df, num_cols, number_range)
    ml_probs, ml_used = train_and_score(training_df, next_draw_df, number_range, MIN_DRAWS_FOR_ML)

    scores = combined_score(freq_counts, poisson_result, ml_probs, ml_used)

    if game_cfg["type"] == "combo" and not ordered:
        recommendation = pick_combo_numbers(scores, pick, number_range)
    else:
        recommendation = pick_digit_numbers(scores, pick, number_range)

    return {
        "n_draws_analyzed": n_draws,
        "ml_used": ml_used,
        "ordered": ordered,
        "poisson_p_value": round(poisson_result["p_value"], 4),
        "significantly_biased": poisson_result["is_significantly_biased"],
        "recommendation": recommendation,
        "top_10_by_score": [int(x) for x in scores.index[:10]],
    }


def analyze_game(game_name, game_cfg, today):
    csv_path = game_cfg["csv"]
    pick = game_cfg["pick"]
    draw_days = game_cfg.get("draw_days")
    is_multi = game_cfg.get("multi_draw_per_day", False)

    if not os.path.exists(csv_path):
        return [{
            "game": game_name,
            "error": f"Walang nahanap na data file: {csv_path}. "
                     f"Maglagay ng CSV (columns: date,n1,...,n{pick}) para masuri ito.",
        }]

    df, num_cols = load_draws(csv_path, pick)

    if not is_multi:
        df = df.tail(MAX_DRAWS_WINDOW).reset_index(drop=True)
        if len(df) < 10:
            return [{
                "game": game_name,
                "error": f"Kulang ang historical data ({len(df)} draws lang). "
                         f"Kailangan ng mas maraming rows sa {csv_path}.",
            }]
        target_date = next_draw_date(draw_days, today)
        try:
            result = run_analysis(df, num_cols, game_cfg)
        except Exception as e:
            return [{"game": game_name, "error": str(e)}]
        result.update({
            "game": game_name,
            "target_draw_date": target_date.isoformat() if target_date else None,
            "target_draw_weekday": target_date.strftime("%A") if target_date else None,
            "is_daily_draw": draw_days is None,
        })
        return [result]

    # multi-draw-per-day games (EZ2, Swertres): hiwalay na analysis per slot
    target_date = next_draw_date(None, today)  # laging bukas para sa daily games
    slots = split_by_slot(df, num_cols)
    out = []
    for slot_label, slot_df in slots.items():
        slot_df = slot_df.tail(MAX_DRAWS_WINDOW).reset_index(drop=True)
        game_label = f"{game_name} ({slot_label})"
        if len(slot_df) < 10:
            out.append({
                "game": game_label,
                "error": f"Kulang ang historical data ({len(slot_df)} draws) para sa slot na ito.",
            })
            continue
        try:
            result = run_analysis(slot_df, num_cols, game_cfg)
        except Exception as e:
            out.append({"game": game_label, "error": str(e)})
            continue
        result.update({
            "game": game_label,
            "target_draw_date": target_date.isoformat() if target_date else None,
            "target_draw_weekday": target_date.strftime("%A") if target_date else None,
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
        except Exception as e:
            game_results = [{"game": game_name, "error": str(e)}]
        for r in game_results:
            results["games"].append(r)
            print(json.dumps(r, indent=2, ensure_ascii=False))

    results["ez2_coverage"] = get_coverage_stats()

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nNasave sa results.json")


if __name__ == "__main__":
    main()
