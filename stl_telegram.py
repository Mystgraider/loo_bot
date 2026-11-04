#!/usr/bin/env python3
"""Send a separate, research-only Aklan STL Pares Telegram signal."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from stl_pares import SLOTS, load_history, rank_pairs, walk_forward_backtest

PH_TZ = ZoneInfo("Asia/Manila")


def build_message(history: list[dict], slot: str) -> str:
    rows = [row for row in history if row["slot"] == slot]
    ranked = rank_pairs(history, slot, 5)
    if not ranked:
        raise ValueError(f"Kulang ang historical data para sa {slot}; kailangan ng hindi bababa sa 20 draws.")

    backtest = walk_forward_backtest(history, slot, min_train=30, top_k=10)
    now = datetime.now(PH_TZ).strftime("%Y-%m-%d %I:%M %p PHT")
    picks = []
    for index, item in enumerate(ranked, start=1):
        a, b = item["pair"]
        picks.append(f"{index}. *{a:02d}-{b:02d}* (historical pair count: {item['observed_pair_count']})")

    if backtest["tested_draws"]:
        test_line = (
            f"Walk-forward Top-10: {backtest['hits']}/{backtest['tested_draws']} exact hits "
            f"({backtest['hit_rate'] * 100:.1f}%); random baseline: "
            f"{backtest['random_baseline_hit_rate'] * 100:.2f}%."
        )
    else:
        test_line = "Walk-forward: kulang pa ang chronological test draws para masukat."

    return "\n".join([
        "🎯 *AKLAN STL PARES — RESEARCH SIGNAL*",
        f"Draw: *{slot}* | Generated: {now}",
        f"Historical draws used: {len(rows)} (slot-specific only)",
        "",
        "*Top 5 ranked pairs:*",
        *picks,
        "",
        test_line,
        "",
        "Ranking ito mula sa historical frequency model, hindi calibrated win probability. "
        "Hindi guarantee ang picks; bawat draw ay random. Research/entertainment only.",
    ])


def main() -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    slot = os.environ.get("STL_SLOT", "").strip()
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.", file=sys.stderr)
        return 2
    if slot not in SLOTS:
        print(f"STL_SLOT must be one of: {', '.join(SLOTS)}", file=sys.stderr)
        return 2

    history = load_history("data/stl_aklan_pares.csv")
    message = build_message(history, slot)
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
        timeout=20,
    )
    if not response.ok:
        raise RuntimeError(f"Telegram API error {response.status_code}: {response.text[:500]}")
    print(f"STL {slot} signal sent successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
