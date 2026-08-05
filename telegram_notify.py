#!/usr/bin/env python3
"""
telegram_notify.py

Nagbabasa ng results.json (galing sa predict.py) at nagpapadala ng
formatted na message sa Telegram gamit ang Bot API.

Kailangan (env vars, ilalagay bilang GitHub Actions secrets):
  TELEGRAM_BOT_TOKEN  - token mula sa @BotFather
  TELEGRAM_CHAT_ID    - chat/channel/group id kung saan ipapadala
"""

import json
import os
import sys

import requests


def format_message(results):
    lines = [f"🎱 *Lotto Statistical Report* — {results['generated_at'][:10]}", ""]
    for g in results["games"]:
        if "error" in g:
            lines.append(f"*{g['game']}*: ⚠️ {g['error']}")
            continue
        nums = ", ".join(f"{n:02d}" for n in g["recommendation"])
        ml_tag = "ML✅" if g["ml_used"] else "ML skipped (kulang pa ang data)"
        lines.append(f"*{g['game']}* ({g['n_draws_analyzed']} draws, {ml_tag})")
        lines.append(f"→ Suggested: `{nums}`")
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
        print(f"Telegram API error ({resp.status_code}): {detail}", file=sys.stderr)
        resp.raise_for_status()
    return resp.json()


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Walang TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID sa environment.", file=sys.stderr)
        sys.exit(1)

    with open("results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    message = format_message(results)
    result = send_telegram_message(token, chat_id, message)
    print("Naipadala sa Telegram:", result.get("ok"))


if __name__ == "__main__":
    main()
