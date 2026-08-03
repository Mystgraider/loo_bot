"""
analyzer.py

Ginagawa nito:
  1. Historical Frequency Analysis  - bilang ng paglabas ng bawat numero
  2. Poisson goodness-of-fit test   - sinusuri kung fair/uniform ang distribution
  3. ML scoring (XGBoost)           - naghahanap ng "pattern" gamit ang rolling
                                      frequency/gap features (weak signal lang,
                                      basahin ang DISCLAIMER sa README)

MAHALAGA: Ang lahat ng ito ay statistical exploration lang. Random at
independent ang bawat draw, kaya ang anumang "score" na lumabas dito ay
HINDI totoong predictive probability ng susunod na draw.
"""

import numpy as np
import pandas as pd
from scipy import stats

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def load_draws(csv_path, pick):
    """CSV format: date,n1,n2,...,n{pick} (isang row per draw, pinakabago sa ibaba o alinmang order;
    pinagbubukod-bukod natin ito by date pagkatapos i-load)."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    num_cols = [c for c in df.columns if c.startswith("n")]
    if len(num_cols) != pick:
        raise ValueError(f"Inaasahan ang {pick} number columns (n1..n{pick}), nakita: {num_cols}")
    return df, num_cols


def frequency_analysis(df, num_cols, number_range):
    lo, hi = number_range
    all_numbers = df[num_cols].values.flatten()
    counts = pd.Series(all_numbers).value_counts().reindex(range(lo, hi + 1), fill_value=0)
    counts = counts.sort_index()
    return counts


def poisson_fairness_test(counts, n_draws, pick, number_range):
    """
    Chi-square goodness-of-fit laban sa Poisson/uniform expectation.
    Null hypothesis: fair/random ang machine (uniform ang tunay na probability
    ng bawat numero). Kung mababa ang p-value (<0.05), may indikasyon ng bias
    -- pero sa audited na lotto machines, bihira itong mangyari, at kahit
    mangyari, hindi pa rin ito sapat para tumpak na ma-predict ang eksaktong
    6-number combination.
    """
    lo, hi = number_range
    n_values = hi - lo + 1
    expected_count = n_draws * pick / n_values  # Poisson lambda per number
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
    """Helper: gumawa ng isang batch ng feature rows (isa per number) gamit ang
    `recent_draws` bilang history. Kung may draw_sets ibinigay, isasama ang
    label (ginagamit para sa training). Kung wala (None), row para sa
    hinaharap na draw na hindi pa alam ang totoong resulta."""
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
    """
    Gumagawa ng DALAWANG bagay:
      1. training_df  - isang row per (draw_index, candidate_number), may label,
         computed gamit LAMANG ang mga draws BAGO sa draw na 'yon (walang leakage)
      2. next_draw_df - isang row per candidate_number, computed gamit ang
         BUONG 500-draw history (kasama ang pinaka-huling known draw) -
         ito ang totoong features na dapat gamitin para i-score ang
         SUSUNOD na draw na hindi pa nangyayari.

    (Naayos: dati, ang ginagamit na "next draw" features ay yung row ng
    pangalawa-hulíng draw na lang, kaya isang draw late ang alam ng model
    kapag nag-predict. Ngayon, gamit na ang kumpletong 500 draws.)
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

    # next_draw features: history = ALL n_draws draws (walang label pa, dahil
    # hindi pa nangyayari ang draw na ito)
    next_rows = _rows_for_index(n_draws, numbers, recent_draws, last_seen, window_sizes, draw_sets=None)
    next_draw_df = pd.DataFrame(next_rows)

    return training_df, next_draw_df


def train_and_score(training_df, next_draw_df, number_range, min_draws_for_ml):
    """
    Nagsasanay ng XGBoost classifier sa LAHAT ng labeled historical rows
    (training_df), tapos gamit ang model na 'yon, i-sco-score ang
    `next_draw_df` (features na batay sa BUONG history, kasama ang
    pinaka-huling known draw) para makuha ang probability estimate ng
    bawat numero para sa SUSUNOD na draw na hindi pa nangyayari.

    Kung kulang ang data o walang xgboost, babalik na lang sa neutral
    score (0.5 for all) - ibig sabihin walang "edge" na nakita.
    """
    lo, hi = number_range
    numbers = np.arange(lo, hi + 1)
    n_draws = training_df["draw_index"].nunique()

    if not HAS_XGB or n_draws < min_draws_for_ml:
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
        random_state=42,  # para consistent/reproducible ang result kada run kung parehong data
    )
    model.fit(training_df[feature_cols], training_df["label"])

    probs = model.predict_proba(next_draw_df[feature_cols])[:, 1]
    return pd.Series(probs, index=next_draw_df["number"].values), True


def combined_score(freq_counts, poisson_result, ml_probs, ml_used, weight_ml=0.5):
    """
    Pinagsasama ang 3 signals sa iisang comparative score (0-1 scale bawat isa):
      - normalized frequency
      - Poisson z-score (deviation mula sa expected)
      - ML predicted probability

    ULITIN: score lang ito para sa "exploration" -- hindi ito totoong
    probability na lalabas ang numerong iyon sa totoong susunod na draw.
    """
    norm_freq = (freq_counts - freq_counts.min()) / (freq_counts.max() - freq_counts.min() + 1e-9)
    z = poisson_result["z_scores"]
    norm_z = (z - z.min()) / (z.max() - z.min() + 1e-9)

    w_ml = weight_ml if ml_used else 0.0
    w_stat = (1 - w_ml) / 2

    score = w_stat * norm_freq + w_stat * norm_z + w_ml * ml_probs.reindex(freq_counts.index).fillna(0.5)
    return score.sort_values(ascending=False)
