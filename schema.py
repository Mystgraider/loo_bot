"""Canonical draw-data schema and validation helpers."""
from __future__ import annotations

import pandas as pd

SLOT_ORDER = {"2:00 PM": 0, "5:00 PM": 1, "9:00 PM": 2}


def canonical_columns(pick: int, multi_draw: bool = False) -> list[str]:
    prefix = ["date", "slot"] if multi_draw else ["date"]
    return prefix + [f"n{i}" for i in range(1, pick + 1)]


def validate_dataframe(
    df: pd.DataFrame,
    pick: int,
    number_range: tuple[int, int],
    *,
    multi_draw: bool = False,
) -> pd.DataFrame:
    """Validate and return a canonical, chronologically sorted dataframe."""
    expected = canonical_columns(pick, multi_draw)
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    extra_n = [c for c in df.columns if c.startswith("n") and c not in expected]
    if extra_n:
        raise ValueError(f"Unexpected number columns: {extra_n}")

    out = df[expected].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("Invalid or missing date value found")

    lo, hi = number_range
    number_cols = [f"n{i}" for i in range(1, pick + 1)]
    for col in number_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"Invalid or missing numeric value in {col}")
        if ((out[col] < lo) | (out[col] > hi)).any():
            raise ValueError(f"Values in {col} must be within {lo}..{hi}")
        out[col] = out[col].astype(int)

    if multi_draw:
        if out["slot"].isna().any() or (out["slot"].astype(str).str.strip() == "").any():
            raise ValueError("Multi-draw data contains a missing slot")
        if not out["slot"].isin(SLOT_ORDER).all():
            bad = sorted(set(out.loc[~out["slot"].isin(SLOT_ORDER), "slot"].astype(str)))
            raise ValueError(f"Unknown draw slot(s): {bad}")
        if out.duplicated(["date", "slot"]).any():
            raise ValueError("Duplicate (date, slot) rows found")
    elif out["date"].duplicated().any():
        raise ValueError("Duplicate draw dates found")

    out["_slot_order"] = out["slot"].map(SLOT_ORDER) if multi_draw else 0
    out = (
        out.sort_values(["date", "_slot_order"], kind="mergesort")
        .drop(columns="_slot_order")
        .reset_index(drop=True)
    )
    return out
