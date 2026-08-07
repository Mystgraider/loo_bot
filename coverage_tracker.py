"""
coverage_tracker.py

Sinusubaybayan ang CUMULATIVE na listahan ng mga natatanging (unique)
EZ2 combination na na-suggest na ng bot sa paglipas ng panahon --
para makita kung ilang porsyento na ng 465 posibleng Rambolito
combinations (walang pakialam sa order) ang na-cover na.

Hindi ito "prediction improvement" -- purong tracking lang ito ng
coverage, katulad ng pag-tick sa isang checklist. Hindi nito
tinataasan ang chance mo sa kahit anong ISANG draw; ipinapakita
lang nito kung gaano karaming magkakaibang combinations na ang
na-try (kung ibabatay mo ang pagbili mo sa mga suggestion ng bot).
"""

import csv
import os
from itertools import combinations

EZ2_RANGE = range(1, 32)  # 1-31
EZ2_TOTAL_COMBOS = len(list(combinations(EZ2_RANGE, 2)))  # 465


def log_combo(combo, log_path="data/ez2_coverage_log.csv"):
    """I-log ang isang (unordered) EZ2 combo kung hindi pa ito naka-log dati.
    combo: iterable ng 2 numero (hal. [7, 22])."""
    key = tuple(sorted(int(x) for x in combo))

    existing = set()
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    existing.add((int(row[0]), int(row[1])))

    if key in existing:
        return False  # wala nang idinagdag, existing na

    write_header_needed = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(key))
    return True


def get_coverage_stats(log_path="data/ez2_coverage_log.csv"):
    if not os.path.exists(log_path):
        return {"unique_combos_covered": 0, "total_possible": EZ2_TOTAL_COMBOS, "percent": 0.0}

    seen = set()
    with open(log_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) == 2:
                seen.add((int(row[0]), int(row[1])))

    pct = round(100 * len(seen) / EZ2_TOTAL_COMBOS, 2)
    return {
        "unique_combos_covered": len(seen),
        "total_possible": EZ2_TOTAL_COMBOS,
        "percent": pct,
    }
