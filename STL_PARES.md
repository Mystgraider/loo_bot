# Aklan STL Pares (research module)

This is an isolated research engine for Aklan STL Pares. A separate Telegram signal sender and pre-draw workflow have now been added on this feature branch; they do not alter or mix with the existing PCSO/lotto Telegram report. The current seed dataset contains 66 valid draw rows transcribed from one independent third-party Aklan history page (34 at 2 PM and 32 at 7 PM). It is not an official operator/PCSO feed and has not been independently authenticated against ticket-office records.

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

## Separate Telegram signal

`stl_telegram.py` formats and sends the top five ordered-pair candidates for one draw slot only. `.github/workflows/stl-pares-telegram.yml` schedules separate messages at 1:50 PM and 6:50 PM Philippine time (10 minutes before the reported 2 PM and 7 PM draws), and supports manual dispatch for either slot. It uses the existing `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` repository secrets. The message includes the number of historical rows used, a chronological Top-10 walk-forward result when available, and an explicit research-only disclaimer. The 2 PM and 7 PM histories are never pooled.

**Activation constraint:** GitHub scheduled workflows run from the repository's default branch. This workflow is currently on the feature branch in draft PR #5, so it will not send scheduled Telegram messages until the PR is reviewed and merged. Do not merge without reviewing the changes. After merge, enable GitHub Actions and verify both repository secrets are present. The existing PCSO workflow and Telegram report remain unchanged.

The current historical source is third-party and not independently authenticated. The current walk-forward sample is too small to establish predictive edge; a Telegram message is a ranked research shortlist, not a reliable prediction or guarantee.
