#!/usr/bin/env python3
"""
scraper.py

Kumukuha ng pinaka-bagong PCSO draw results mula sa lottopcso.com --
isang independent/unofficial na site (hindi opisyal na PCSO source),
pero regular itong na-a-update at may kumpletong history table per
laro. Wala kasing libreng public API ang PCSO mismo na puwedeng
i-access nang automated.

Nag-a-a-APPEND lang ng BAGONG draws (base sa petsa, at sa mga
multi-draw-per-day na laro, base rin sa oras) -- hindi na dinadaan
pa ang mga row na existing na sa CSV.

PAALALA: third-party site scraping ito, hindi opisyal na PCSO API.
Puwedeng magbago ang HTML structure ng site anumang oras. Kung
mag-fail o mag-warning ang script na "hindi mahanap ang history
table," i-check muna ang structure ng target URL sa browser bago
mag-debug ng code.
"""

import csv
import os
import re
import sys
from datetime import datetime

import requests
from bs4 import BeautifulSoup

BASE = "https://www.lottopcso.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; lotto-stat-bot/1.0)"}

# combo games (6 unique numbers, walang ulit): key -> (url slug, csv path, pick)
COMBO_GAMES = {
    "6_55": ("6-55-lotto-result-history-and-summary", "data/draws_6_55.csv", 6),
    "6_58": ("6-58-lotto-result-history-and-summary", "data/draws_6_58.csv", 6),
    "6_49": ("6-49-lotto-result-history-and-summary", "data/draws_6_49.csv", 6),
    "6_45": ("6-45-lotto-result-history-and-summary", "data/draws_6_45.csv", 6),
    "6_42": ("6-42-lotto-result-history-and-summary", "data/draws_6_42.csv", 6),
}

# digit games na minsan lang sa isang araw ang draw (9PM lang): key -> (url slug, csv path, pick)
SINGLE_DRAW_DIGIT_GAMES = {
    "6d": ("6d-lotto-results-6d-history-and-summary", "data/draws_6d.csv", 6),
    "4d": ("4d-lotto-results-4d-history-and-summary", "data/draws_4d.csv", 4),
}

# digit games na 3x sa isang araw ang draw (2PM, 5PM, 9PM): key -> (url slug, csv path, pick)
MULTI_DRAW_DIGIT_GAMES = {
    "swertres": ("swertres-results-today-history-and-summary", "data/draws_swertres.csv", 3),
    "ez2": ("ez2-result-today-lotto-history-and-summary", "data/draws_ez2.csv", 2),
}


def fetch_soup(slug):
    url = f"{BASE}/{slug}/"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def parse_date(text):
    text = text.strip()
    text = re.sub(r"[\[\]]", "", text)
    text = text.replace("Mar ", "March ").replace("Aug. ", "August ")
    for fmt in ("%b. %d, %Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def looks_like_pick_numbers(text, pick):
    nums = re.findall(r"\d+", text)
    return len(nums) == pick


def find_history_table(soup, header1_kw, pick_fallback=None):
    """Hanapin ang tamang table: unang column 'Draw Date', pangalawa
    ay tumutugma sa header1_kw. Kung walang match at may pick_fallback,
    gamitin na lang ang unang table na ang unang data row ay may
    tamang bilang ng numero sa ikalawang column."""
    tables = soup.find_all("table")
    for table in tables:
        header_row = table.find("tr")
        if not header_row:
            continue
        headers = [c.get_text(strip=True).lower() for c in header_row.find_all(["th", "td"])]
        if len(headers) < 2 or "draw date" not in headers[0]:
            continue
        if header1_kw and header1_kw in headers[1]:
            return table

    if pick_fallback:
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            header_cells = [c.get_text(strip=True).lower() for c in rows[0].find_all(["th", "td"])]
            if not header_cells or "draw date" not in header_cells[0]:
                continue
            data_cells = rows[1].find_all(["td", "th"])
            if len(data_cells) >= 2 and looks_like_pick_numbers(data_cells[1].get_text(strip=True), pick_fallback):
                return table
    return None


def existing_dates_combo(csv_path):
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {row["date"] for row in csv.DictReader(f)}


def existing_date_combo_pairs(csv_path, pick):
    if not os.path.exists(csv_path):
        return set()
    pairs = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            combo = tuple(row[f"n{i+1}"] for i in range(pick))
            pairs.add((row["date"], combo))
    return pairs


def append_rows(csv_path, pick, rows):
    if not rows:
        return
    rows.sort(key=lambda r: r[0])
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["date"] + [f"n{i+1}" for i in range(pick)])
        for date_str, nums in rows:
            w.writerow([date_str] + nums)


def scrape_single_draw_game(key, slug, csv_path, pick):
    soup = fetch_soup(slug)
    table = find_history_table(soup, "winning number", pick_fallback=pick)
    if table is None:
        print(f"[{key}] WARNING: hindi mahanap ang history table -- baka nagbago ang site structure.")
        return []

    existing = existing_dates_combo(csv_path)
    new_rows = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        d = parse_date(cells[0].get_text(strip=True))
        combo_text = cells[1].get_text(strip=True)
        if d is None or combo_text in ("-", "\u2013", ""):
            continue
        nums = [int(x) for x in re.findall(r"\d+", combo_text)]
        if len(nums) != pick:
            continue
        date_str = d.isoformat()
        if date_str in existing:
            continue
        new_rows.append((date_str, nums))
        existing.add(date_str)

    append_rows(csv_path, pick, new_rows)
    return new_rows


def scrape_multi_draw_digit_game(key, slug, csv_path, pick):
    soup = fetch_soup(slug)
    table = find_history_table(soup, "2:00 pm")
    if table is None:
        print(f"[{key}] WARNING: hindi mahanap ang history table -- baka nagbago ang site structure.")
        return []

    existing = existing_date_combo_pairs(csv_path, pick)
    new_rows = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        d = parse_date(cells[0].get_text(strip=True))
        if d is None:
            continue
        date_str = d.isoformat()
        for draw_cell in cells[1:4]:
            combo_text = draw_cell.get_text(strip=True)
            if combo_text in ("-", "\u2013", ""):
                continue
            nums = [int(x) for x in re.findall(r"\d+", combo_text)]
            if len(nums) != pick:
                continue
            combo = tuple(str(n) for n in nums)
            if (date_str, combo) in existing:
                continue
            new_rows.append((date_str, nums))
            existing.add((date_str, combo))

    append_rows(csv_path, pick, new_rows)
    return new_rows


def main():
    total_new = 0

    for key, (slug, csv_path, pick) in COMBO_GAMES.items():
        try:
            new_rows = scrape_single_draw_game(key, slug, csv_path, pick)
            print(f"[{key}] +{len(new_rows)} bagong draws")
            total_new += len(new_rows)
        except Exception as e:
            print(f"[{key}] ERROR: {e}", file=sys.stderr)

    for key, (slug, csv_path, pick) in SINGLE_DRAW_DIGIT_GAMES.items():
        try:
            new_rows = scrape_single_draw_game(key, slug, csv_path, pick)
            print(f"[{key}] +{len(new_rows)} bagong draws")
            total_new += len(new_rows)
        except Exception as e:
            print(f"[{key}] ERROR: {e}", file=sys.stderr)

    for key, (slug, csv_path, pick) in MULTI_DRAW_DIGIT_GAMES.items():
        try:
            new_rows = scrape_multi_draw_digit_game(key, slug, csv_path, pick)
            print(f"[{key}] +{len(new_rows)} bagong draws")
            total_new += len(new_rows)
        except Exception as e:
            print(f"[{key}] ERROR: {e}", file=sys.stderr)

    print(f"\nTotal bagong draws na na-add sa lahat ng laro: {total_new}")

    # ginagamit ito ng GitHub Actions workflow para malaman kung may
    # ia-commit pa (skip commit kung walang bagong data)
    with open("scrape_summary.txt", "w") as f:
        f.write(str(total_new))


if __name__ == "__main__":
    main()
