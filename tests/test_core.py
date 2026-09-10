import pandas as pd
import pytest

from config import GAMES
from predict import pick_digit_numbers
from schema import validate_dataframe
from scraper import parse_date
from backtest_v2 import exact_hit, random_expected, random_prediction


def test_parse_date_accepts_iso_csv_date():
    assert parse_date("2026-09-09").isoformat() == "2026-09-09"


def test_multi_draw_schema_requires_explicit_slot():
    df = pd.DataFrame({"date": ["2026-09-09"], "n1": [1], "n2": [2]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df, 2, (1, 31), multi_draw=True)


def test_multi_draw_schema_sorts_by_slot():
    df = pd.DataFrame({
        "date": ["2026-09-09", "2026-09-09", "2026-09-09"],
        "slot": ["9:00 PM", "2:00 PM", "5:00 PM"],
        "n1": [9, 1, 5],
        "n2": [8, 2, 6],
    })
    out = validate_dataframe(df, 2, (1, 31), multi_draw=True)
    assert out["slot"].tolist() == ["2:00 PM", "5:00 PM", "9:00 PM"]


def test_digit_picker_allows_repetition():
    cfg = GAMES["swertres"]
    scores = pd.Series([1.0] * 10, index=range(10))
    picked = pick_digit_numbers(scores, 6, cfg["range"], seed=7)
    assert len(picked) == 6
    assert all(0 <= x <= 9 for x in picked)


def test_combo_config_disallows_repetition():
    assert GAMES["6/55"]["allow_repetition"] is False
    assert GAMES["swertres"]["allow_repetition"] is True


def test_exact_hit_respects_order():
    assert exact_hit([1, 2], [1, 2], True)
    assert not exact_hit([2, 1], [1, 2], True)
    assert exact_hit([2, 1], [1, 2], False)


def test_random_prediction_respects_game_semantics():
    combo = GAMES["6/55"]
    pred = random_prediction(combo["range"], combo["pick"], combo, seed=1)
    assert len(pred) == 6
    assert len(set(pred)) == 6
    assert pred == sorted(pred)

    digits = GAMES["swertres"]
    pred_digits = random_prediction(digits["range"], digits["pick"], digits, seed=1)
    assert len(pred_digits) == 3
    assert all(0 <= x <= 9 for x in pred_digits)


def test_random_exact_probability():
    combo = GAMES["6/55"]
    _, p = random_expected(combo["range"], combo["pick"], combo)
    assert p == pytest.approx(1 / 20358520)

    ez2 = GAMES["ez2"]
    _, p_ez2 = random_expected(ez2["range"], ez2["pick"], ez2)
    assert p_ez2 == pytest.approx(1 / (31 * 30))

    swertres = GAMES["swertres"]
    _, p_sw = random_expected(swertres["range"], swertres["pick"], swertres)
    assert p_sw == pytest.approx(1 / 1000)
