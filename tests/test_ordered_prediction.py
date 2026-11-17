import pandas as pd

from predict import pick_ordered_numbers


def test_ordered_picker_uses_each_position_independently():
    df = pd.DataFrame({
        "n1": [7, 7, 2, 7, 2],
        "n2": [3, 8, 8, 8, 3],
        "n3": [9, 4, 4, 9, 4],
    })

    recommendation, position_top = pick_ordered_numbers(
        df,
        ["n1", "n2", "n3"],
        pick=3,
        number_range=(0, 9),
        window=120,
    )

    assert recommendation == [7, 8, 4]
    assert position_top[0][0] == 7
    assert position_top[1][0] == 8
    assert position_top[2][0] == 4


def test_ordered_picker_preserves_repeated_digits_and_order():
    df = pd.DataFrame({
        "n1": [5, 5, 2],
        "n2": [5, 3, 3],
        "n3": [1, 1, 9],
    })

    recommendation, _ = pick_ordered_numbers(
        df,
        ["n1", "n2", "n3"],
        pick=3,
        number_range=(0, 9),
    )

    assert recommendation == [5, 3, 1]
    assert len(recommendation) == 3
    assert all(0 <= digit <= 9 for digit in recommendation)


def test_ordered_picker_uses_only_latest_window():
    df = pd.DataFrame({
        "n1": [9, 1, 1, 1],
        "n2": [9, 2, 2, 2],
    })

    recommendation, _ = pick_ordered_numbers(
        df,
        ["n1", "n2"],
        pick=2,
        number_range=(0, 9),
        window=3,
    )

    assert recommendation == [1, 2]
