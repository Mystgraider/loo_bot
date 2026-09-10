#!/usr/bin/env python3
"""Fetch PCSO draw history and rewrite each CSV in canonical form.

Source is lottopcso.com (third-party/unofficial). The scraper fails the run if
any configured game cannot be fetched or parsed, preventing stale-data reports.
Multi-draw games store an explicit ``slot`` column; old CSVs without it are
migrated using their existing per-date row order once, then canonicalized.
"""

import csv
import json
import os
import re
import sys
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from config import GAMES, SLOT_LABELS
from schema import SLOT_ORDER, canonical_columns, validate_dataframe

BASE = "https://www.lottopcso.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; lotto-stat-bot/1.1)"}

COMBO_GAMES = {
    "6_55": ("6-55-lotto-result-history-and-summary", "data/draws_6_55.csv", 6),
    "6_58": ("6-58-lotto-result-history-and-summary", "data/draws_6_58.csv", 6),
    "6_49": ("6-49-lotto-result-history-and-summary", "data/draws_6_49.csv", 6),
    "6_45": ("6-45-lotto-result-history-and-summary", "data/draws_6_45.csv", 6),
    "6_42": ("6-42-lotto-result-history-and-summary", "data/draws_6_42.csv", 6),
}
SINGLE_DRAW_DIGIT_GAMES = {
    "6d": ("6d-lotto-results-6d-history-and-summary", "data/draws_6d.csv", 6),
    "4d": ("4d-lotto-results-4d-history-and-summary", "data/draws_4d.csv", 4),
}
MULTI_DRAW_DIGIT_GAMES = {
    "swertres": ("swertres-results-today-history-and-summary", "data/draws_swertres.csv", 3),
    "ez2": ("ez2-result-today-lotto-history-and-summary", "data/draws_ez2.csv", 2),
}


def fetch_soup(slug):
    resp = requests.get(f"{BASE}/{slug}/", headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_date(text):
    text = re.sub(r"[\[\]]", "", text.strip())
    text = text.replace("Mar ", "March ").replace("Aug. ", "August ")
    for fmt in ("%b. %d, %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def find_history_table(soup, header1_kw, pick_fallback=None):
    tables = soup.find_all("table")
    for table in tables:
        header_row = table.find("tr")
        if not header_row:
            continue
        headers = [c.get_text(strip=True).lower() for c in header_row.find_all(["th", "td"])]
        if len(headers) >= 2 and "draw date" in headers[0] and header1_kw in headers[1]:
            return table
    if pick_fallback:
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            headers = [c.get_text(strip=True).lower() for c in rows[0].find_all(["th", "td"])]
            if not headers or "draw date" not in headers[0]:
                continue
            cells = rows[1].find_all(["td", "th"])
            if len(cells) >= 2 and len(re.findall(r"\d+", cells[1].get_text(strip=True))) == pick_fallback:
                return table
    return None


def parse_existing(csv_path, pick, multi_draw):
    """Read current CSV, migrating legacy multi-draw rows without ``slot``."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    out = []
    for row in rows:
        try:
            d = parse_date(str(row["date"]))
            nums = [int(row[f"n{i}"]) for i in range(1, pick + 1)]
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Invalid existing row in {csv_path}")
        if d is None:
            raise ValueError(f"Invalid date in {csv_path}")
        item = {"date": d.isoformat(), "nums": nums}
        if multi_draw:
            item["slot"] = row.get("slot", "").strip()
        out.append(item)

    if multi_draw and any(not r.get("slot") for r in out):
        # Legacy migration: the old format encoded 2PM->5PM->9PM by row order.
        # Stable sorting preserves that historical order within each date.
        out.sort(key=lambda r: r["date"])
        counters = {}
        for row in out:
            idx = counters.get(row["date"], 0)
            if idx >= len(SLOT_LABELS):
                raise ValueError(f"More than {len(SLOT_LABELS)} legacy rows for {row['date']}")
            row["slot"] = SLOT_LABELS[idx]
            counters[row["date"]] = idx + 1
    return out


def scrape_single(slug, csv_path, pick, game_cfg):
    soup = fetch_soup(slug)
    table = find_history_table(soup, "winning number", pick_fallback=pick)
    if table is None:
        raise RuntimeError("History table not found")

    by_key = {(r["date"], ""): r for r in parse_existing(csv_path, pick, False)}
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        d = parse_date(cells[0].get_text(strip=True))
        nums = [int(x) for x in re.findall(r"\d+", cells[1].get_text(strip=True))]
        if d is None or len(nums) != pick:
            continue
        key = (d.isoformat(), "")
        by_key[key] = {"date": d.isoformat(), "nums": nums}

    rows = list(by_key.values())
    df = _rows_to_df(rows, pick, False)
    df = validate_dataframe(df, pick, tuple(game_cfg["range"]), multi_draw=False)
    _write_df(csv_path, df, pick, False)
    return len(rows)


def scrape_multi(slug, csv_path, pick, game_cfg):
    soup = fetch_soup(slug)
    table = find_history_table(soup, "2:00 pm")
    if table is None:
        raise RuntimeError("Multi-draw history table not found")

    existing = {(r["date"], r["slot"]): r for r in parse_existing(csv_path, pick, True)}
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        d = parse_date(cells[0].get_text(strip=True))
        if d is None:
            continue
        for idx, draw_cell in enumerate(cells[1:4]):
            text = draw_cell.get_text(strip=True)
            if text in ("-", "\u2013", ""):
                continue
            nums = [int(x) for x in re.findall(r"\d+", text)]
            if len(nums) != pick:
                continue
            slot = SLOT_LABELS[idx]
            existing[(d.isoformat(), slot)] = {"date": d.isoformat(), "slot": slot, "nums": nums}

    rows = list(existing.values())
    df = _rows_to_df(rows, pick, True)
    df = validate_dataframe(df, pick, tuple(game_cfg["range"]), multi_draw=True)
    _write_df(csv_path, df, pick, True)
    return len(rows)


def _rows_to_df(rows, pick, multi_draw):
    columns = canonical_columns(pick, multi_draw)
    data = []
    for row in rows:
        values = [row["date"]]
        if multi_draw:
            values.append(row["slot"])
        values.extend(row["nums"])
        data.append(values)
    import pandas as pd
    return pd.DataFrame(data, columns=columns)


def _write_df(csv_path, df, pick, multi_draw):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False, columns=canonical_columns(pick, multi_draw))


def main():
    failures = []
    summary = {}

    for key, (slug, path, pick) in COMBO_GAMES.items():
        try:
            summary[key] = scrape_single(slug, path, pick, GAMES[key.replace("_", "/")])
            print(f"[{key}] canonical rows: {summary[key]}")
        except Exception as exc:
            failures.append(f"{key}: {exc}")
            print(f"[{key}] ERROR: {exc}", file=sys.stderr)

    for key, (slug, path, pick) in SINGLE_DRAW_DIGIT_GAMES.items():
        try:
            summary[key] = scrape_single(slug, path, pick, GAMES[key])
            print(f"[{key}] canonical rows: {summary[key]}")
        except Exception as exc:
            failures.append(f"{key}: {exc}")
            print(f"[{key}] ERROR: {exc}", file=sys.stderr)

    for key, (slug, path, pick) in MULTI_DRAW_DIGIT_GAMES.items():
        try:
            summary[key] = scrape_multi(slug, path, pick, GAMES[key])
            print(f"[{key}] canonical rows: {summary[key]}")
        except Exception as exc:
            failures.append(f"{key}: {exc}")
            print(f"[{key}] ERROR: {exc}", file=sys.stderr)

    with open("scrape_summary.json", "w", encoding="utf-8") as f:
        json.dump({"games": summary, "failures": failures}, f, indent=2)

    if failures:
        raise SystemExit("Scrape failed for one or more games: " + "; ".join(failures))

    print(f"\nScrape/validation complete: {len(summary)} games healthy.")


if __name__ == "__main__":
    main()
