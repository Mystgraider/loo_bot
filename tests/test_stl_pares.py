import csv
from pathlib import Path

import pytest

from stl_pares import SLOTS, build_report, load_history, rank_pairs, walk_forward_backtest


def make_rows(count=80, slot="2:00 PM"):
    # Deterministic synthetic data is used only to test code mechanics.
    return [{"date": f"2026-01-{i + 1:02d}", "slot": slot,
             "n1": 7 if i % 3 else 8, "n2": 12 if i % 4 else 13,
             "source": "synthetic-test"} for i in range(count)]


def test_load_history_validates_and_sorts(tmp_path):
    path = tmp_path / "stl.csv"
    path.write_text("date,slot,n1,n2,source\n2026-01-02,7:00 PM,2,3,test\n2026-01-01,2:00 PM,1,40,test\n")
    rows = load_history(path)
    assert rows[0]["date"] == "2026-01-01"
    assert rows[1]["slot"] == "7:00 PM"


@pytest.mark.parametrize("slot,n1,n2", [
    ("9:00 PM", 1, 2), ("2:00 PM", 0, 2), ("7:00 PM", 1, 41)
])
def test_load_history_rejects_invalid_rows(tmp_path, slot, n1, n2):
    path = tmp_path / "stl.csv"
    path.write_text(f"date,slot,n1,n2\n2026-01-01,{slot},{n1},{n2}\n")
    with pytest.raises(ValueError):
        load_history(path)


def test_ranker_requires_minimum_history():
    assert rank_pairs(make_rows(19), "2:00 PM") == []
    assert len(rank_pairs(make_rows(20), "2:00 PM", 5)) == 5


def test_walk_forward_is_chronological_and_reports_baseline():
    rows = make_rows(60)
    result = walk_forward_backtest(rows, "2:00 PM", min_train=30, top_k=10)
    assert result["tested_draws"] == 30
    assert result["random_baseline_hit_rate"] == 10 / 1600
    assert result["status"] == "EVALUATED"


def test_slots_are_evaluated_independently():
    rows = make_rows(40, "2:00 PM") + make_rows(40, "7:00 PM")
    report = build_report(rows)
    assert set(report["slots"]) == set(SLOTS)
    assert report["slots"]["2:00 PM"]["draws"] == 40
    assert report["slots"]["7:00 PM"]["draws"] == 40


def test_repository_seed_dataset_is_valid_and_slot_separated():
    data_path = Path(__file__).resolve().parents[1] / "data" / "stl_aklan_pares.csv"
    rows = load_history(data_path)
    assert len(rows) == 65
    assert sum(row["slot"] == "2:00 PM" for row in rows) == 33
    assert sum(row["slot"] == "7:00 PM" for row in rows) == 32
    assert len({(row["date"], row["slot"]) for row in rows}) == len(rows)


def test_repository_seed_backtest_does_not_claim_edge_without_evidence():
    data_path = Path(__file__).resolve().parents[1] / "data" / "stl_aklan_pares.csv"
    rows = load_history(data_path)
    report = build_report(rows)
    assert report["status"] == "RESEARCH_ONLY"
    for slot in SLOTS:
        assert report["slots"][slot]["backtest"]["status"] in {
            "EVALUATED", "INSUFFICIENT_HISTORY"
        }
