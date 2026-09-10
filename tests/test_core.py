import pandas as pd
import pytest

from config import GAMES
from predict import pick_digit_numbers
from schema import validate_dataframe


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
