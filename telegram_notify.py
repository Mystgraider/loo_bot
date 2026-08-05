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
        # RuntimeError na may klarong detalye (hindi lang generic HTTP status)
        raise RuntimeError(f"Telegram API error ({resp.status_code}): {detail}")
    return resp.json()


def write_debug_file(token, chat_id, error_text):
    """Isulat ang diagnostic info sa isang file na puwedeng i-commit at
    tingnan (HINDI kasama ang buong token -- masked lang, para safe)."""
    masked_token = f"{token[:6]}...{token[-4:]}" if token and len(token) > 12 else "(sobrang ikli o wala)"
    with open("telegram_debug.txt", "w", encoding="utf-8") as f:
        f.write("=== Telegram send debug info ===\n")
        f.write(f"chat_id ginamit: {chat_id!r}\n")
        f.write(f"bot token (masked): {masked_token}\n")
        f.write(f"token length: {len(token) if token else 0}\n")
        f.write(f"error: {error_text}\n")


def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Walang TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID sa environment.", file=sys.stderr)
        sys.exit(1)

    with open("results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    message = format_message(results)

    try:
        result = send_telegram_message(token, chat_id, message)
        print("Naipadala sa Telegram:", result.get("ok"))
        # tanggalin ang lumang debug file kung successful na (para malinis)
        if os.path.exists("telegram_debug.txt"):
            os.remove("telegram_debug.txt")
    except Exception as e:
        print(str(e), file=sys.stderr)
        write_debug_file(token, chat_id, str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
