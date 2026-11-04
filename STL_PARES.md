# Aklan STL Pares (research module)

This is an isolated research engine for Aklan STL Pares. It is **not enabled as a live Telegram signal**. The current seed dataset contains 66 valid draw rows transcribed from one independent third-party Aklan history page (34 at 2 PM and 32 at 7 PM). It is not an official operator/PCSO feed and has not been independently authenticated against ticket-office records.

## Data contract

Create `data/stl_aklan_pares.csv` with these columns:

```csv
date,slot,n1,n2,source
2026-01-01,2:00 PM,12,34,verified-source-name
2026-01-01,7:00 PM,5,21,verified-source-name
```

- Dates are ISO `YYYY-MM-DD`.
- Slots are exactly `2:00 PM` or `7:00 PM`.
- Each number must be 01 through 40.
- One row per date and slot. Preserve draw order in `n1,n2`.
- Do not seed this file from predictions or synthetic test data.
- The current seed source is documented in each row. It is an independent aggregator, not an official feed.
- Five malformed/ambiguous compact results were quarantined in `data/stl_aklan_pares_rejected.csv`; they are excluded from analysis rather than guessed.
- Missing draws are omitted, not treated as zero or fabricated.

## Run

```bash
python stl_pares.py --csv data/stl_aklan_pares.csv --output stl_pares_results.json
python -m pytest -q
```

## Method and validation

The experimental ranker combines exact ordered-pair frequency with smoothed position frequencies. It is a candidate-ranking heuristic, not a probability guarantee. The walk-forward backtest uses an expanding chronological training window and reports top-10 exact-order hits against the uniform random baseline (10/1600 per draw).

The current sample is too small for a credible edge claim: with a 30-draw minimum training window, the 65-row dataset yields only a handful of walk-forward test draws per slot. A credible edge requires substantially more verified draws, a pre-registered evaluation window, uncertainty intervals, and repeated out-of-sample performance. Do not tune weights on the same holdout used for final evaluation. Reversed pairs are not counted as exact hits. A high score is a ranking score, not a calibrated probability.

## Deployment status

The existing PCSO workflow and Telegram report remain unchanged. The existing scheduled workflow runs at 9:30 PM Philippine time, after the proposed STL draw times, so it cannot deliver actionable same-day STL picks. Add pre-draw schedules only after the historical source, timezone, schedule, and data integrity checks are confirmed. The current data source reports Aklan Pares draws at 2:00 PM and 7:00 PM PHT; confirm locally before operational use.
