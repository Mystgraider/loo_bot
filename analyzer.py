"""Statistical and ML analysis primitives used by production and backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from schema import validate_dataframe

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def load_draws(csv_path, pick, number_range=None, *, multi_draw=False):
    """Load, validate, and canonically sort a draw CSV."""
    df = pd.read_csv(csv_path)
    if number_range is None:
        # Kept for compatibility with callers that only need structural loading.
        lo, hi = 0, 10**9
    else:
        lo, hi = number_range
    df = validate_dataframe(df, pick, (lo, hi), multi_draw=multi_draw)
    num_cols = [f"n{i}" for i in range(1, pick + 1)]
    return df, num_cols


def frequency_analysis(df, num_cols, number_range):
    lo, hi = number_range
    all_numbers = df[num_cols].values.flatten()
    counts = pd.Series(all_numbers).value_counts().reindex(range(lo, hi + 1), fill_value=0)
    return counts.sort_index()


def poisson_fairness_test(counts, n_draws, pick, number_range):
    lo, hi = number_range
    n_values = hi - lo + 1
    expected_count = n_draws * pick / n_values
    expected = np.full(n_values, expected_count)
    chi2_stat, p_value = stats.chisquare(f_obs=counts.values, f_exp=expected)
    z_scores = (counts.values - expected_count) / np.sqrt(expected_count)
    return {
        "expected_count_per_number": expected_count,
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "z_scores": pd.Series(z_scores, index=counts.index),
        "is_significantly_biased": bool(p_value < 0.05),
    }


def _rows_for_index(t, numbers, recent_draws, last_seen, window_sizes, draw_sets=None):
    out = []
    for n in numbers:
        feat = {"number": n}
        for w in window_sizes:
            window = recent_draws[-w:] if recent_draws else []
            feat[f"freq_last_{w}"] = sum(1 for d in window if n in d)
        feat["gap_since_last_seen"] = (t - last_seen[n]) if last_seen[n] >= 0 else t
        feat["overall_freq_so_far"] = sum(1 for d in recent_draws if n in d)
        feat["draw_index"] = t
        if draw_sets is not None:
            feat["label"] = 1 if n in draw_sets[t] else 0
        out.append(feat)
    return out


def build_ml_features(df, num_cols, number_range, window_sizes=(10, 30, 100)):
    """Build leakage-safe historical rows plus a true next-draw feature frame.

    For digit games, repeated digits remain valid because the label is based on
    digit presence rather than converting the entire draw into a unique-number
    combination. Position-specific modeling is intentionally a later phase.
    """
    lo, hi = number_range
    numbers = np.arange(lo, hi + 1)
    n_draws = len(df)
    draw_sets = [set(row) for row in df[num_cols].values.tolist()]

    rows = []
    last_seen = {n: -1 for n in numbers}
    recent_draws = []
    for t in range(n_draws):
        rows.extend(_rows_for_index(t, numbers, recent_draws, last_seen, window_sizes, draw_sets))
        for n in draw_sets[t]:
            last_seen[n] = t
        recent_draws.append(draw_sets[t])

    training_df = pd.DataFrame(rows)
    next_draw_df = pd.DataFrame(
        _rows_for_index(n_draws, numbers, recent_draws, last_seen, window_sizes)
    )
    return training_df, next_draw_df


def train_and_score(training_df, next_draw_df, number_range, min_draws_for_ml):
    lo, hi = number_range
    numbers = np.arange(lo, hi + 1)
    if not HAS_XGB or training_df.empty:
        return pd.Series(0.5, index=numbers), False

    n_draws = training_df["draw_index"].nunique()
    if n_draws < min_draws_for_ml:
        return pd.Series(0.5, index=numbers), False

    feature_cols = [c for c in training_df.columns if c not in ("label", "draw_index", "number")]
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(training_df[feature_cols], training_df["label"].astype(int))
    probs = model.predict_proba(next_draw_df[feature_cols])[:, 1]
    return pd.Series(probs, index=next_draw_df["number"].astype(int)), True


def combined_score(freq_counts, poisson_result, ml_probs, ml_used, weight_ml=0.5):
    norm_freq = (freq_counts - freq_counts.min()) / (freq_counts.max() - freq_counts.min() + 1e-9)
    z = poisson_result["z_scores"]
    norm_z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    w_ml = weight_ml if ml_used else 0.0
    w_stat = (1 - w_ml) / 2
    score = (
        w_stat * norm_freq
        + w_stat * norm_z
        + w_ml * ml_probs.reindex(freq_counts.index).fillna(0.5)
    )
    return score.sort_values(ascending=False)
