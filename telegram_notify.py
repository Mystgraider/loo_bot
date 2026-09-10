#!/usr/bin/env python3
"""Format results.json and send the daily report to Telegram."""

import json
import os
import sys

import requests


def format_message(results):
    lines = ["🎱 *Lotto Statistical Report*", ""]
    for g in results["games"]:
        if "error" in g:
            lines.append(f"*{g['game']}*: ⚠️ {g['error']}")
            continue
        nums = ", ".join(f"{n:02d}" for n in g["recommendation"])
        ml_tag = "ML✅" if g["ml_used"] else "ML skipped (kulang pa ang data)"
        if g.get("target_draw_time"):
            date_label = f"para sa {g['target_draw_date']}, {g['target_draw_time']}"
        else:
            date_label = f"para sa {g['target_draw_date']} ({g.get('target_draw_weekday', '')})"
        order_note = " -- *may order*" if g.get("ordered") else ""
        lines.append(f"*{g['game']}* -- {date_label}")
        lines.append(f"({g['n_draws_analyzed']} draws, {ml_tag})")
        lines.append(f"→ Suggested: `{nums}`{order_note}")
        lines.append("")

    cov = results.get("ez2_coverage")
    if cov:
        lines.append(
            f"📊 *EZ2 Coverage:* {cov['unique_combos_covered']}/{cov['total_possible']} "
            f"({cov['percent']}%) ng posibleng Rambolito combinations na na-suggest na"
        )
        lines.append("")
    lines.append(results["disclaimer"])
    return "\n".join(lines)


def send_telegram_message(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = requests.post(
        url,
        data={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=15,
    )
    if not resp.ok:
        try:
            detail = resp.json().get("description", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"Telegram API error ({resp.status_code}): {detail}")
    return resp.json()


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Walang TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID sa environment.", file=sys.stderr)
        return 1

    with open("results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    message = format_message(results)

    try:
        result = send_telegram_message(token, chat_id, message)
        print("Naipadala sa Telegram:", result.get("ok"))
        return 0
    except Exception as exc:
        # Never write credentials or chat IDs to a tracked repository file.
        print(f"Telegram send failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
