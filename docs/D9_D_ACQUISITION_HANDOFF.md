# D9-D Acquisition Handoff — 12-issuer SEC EDGAR filings + prices

**Status as of 2026-09-17**: all D9-D AUTHORITATIVE (v2) code and tests are
complete and passing on synthetic fixtures. The real 12-issuer SEC EDGAR
10-K/10-Q/8-K corpus has **not** been acquired — this is the one remaining
step before `formation_dev`/`validation`/`historical_evaluation` can run on
real data and the required report can be written.

## Why this is a handoff, not a run

This session's own egress proxy blocks both SEC hosts:

```
$ curl -sS --max-time 10 https://data.sec.gov/submissions/CIK0000320193.json
curl: (56) CONNECT tunnel failed, response 403
```

Confirmed for both `data.sec.gov` and `www.sec.gov` (organization policy,
not a transient failure). Yahoo Finance access was not separately
re-tested in this session but the same acquisition script's price leg
worked previously for the v1 dataset in a prior working session.

## What to run

From an environment with real network access to SEC EDGAR and Yahoo
Finance, with this repository checked out at this branch:

```bash
pip install -e ".[dev]"   # picks up pandas-market-calendars, pysentiment2, yfinance et al.
python scripts/acquire_sec_filings_12issuer_daily.py
```

This acquires and freezes two datasets:

- `data/manifests/sec_filings_12issuer_2015_2025_v1.json` +
  `data/raw/sec_filings_12issuer_2015_2025_v1/` (gitignored, do not commit)
- `data/manifests/yf_sentiment_equities_daily_2015_2025_v1.json` +
  `data/raw/yf_sentiment_equities_daily_2015_2025_v1/` (gitignored)

Flags: `--no-freeze` for a validate-only dry run, `--max-filings-per-symbol N`
to sanity-check on a small slice first, `--raw-dir PATH` to redirect output.

## Before running: verify the CIK map

`scripts/acquire_sec_filings_12issuer_daily.py`'s `SYMBOL_CIK` maps all 12
issuers to their SEC Central Index Keys. **AAPL/MSFT/AMZN match this repo's
already-frozen v1 acquisition exactly** (cross-checked against
`data/manifests/edgar_8k_yf_megacap_daily_2015_2025_v1.json`). The other
nine (GOOGL, NVDA, JPM, XOM, JNJ, PG, WMT, HD, KO) were **not**
independently verified against a live SEC source when this script was
written (network was blocked in that session too). Before running for
real, confirm each against:

```
https://data.sec.gov/submissions/CIK##########.json
```

(zero-padded to 10 digits) and check the returned `name`/`tickers` fields
match. A wrong CIK would silently acquire the wrong company's filings.

## What comes back

Hand back (or push to this branch):

1. `data/manifests/sec_filings_12issuer_2015_2025_v1.json`
2. `data/manifests/yf_sentiment_equities_daily_2015_2025_v1.json`
3. The full `data/raw/sec_filings_12issuer_2015_2025_v1/` and
   `data/raw/yf_sentiment_equities_daily_2015_2025_v1/` trees (needed
   locally to run the study — never committed, per this repo's own
   `data/raw/` gitignore convention)

## What happens after data lands

With `data/raw/` populated:

1. Load filing events from `sec_filings_12issuer_2015_2025_v1`'s index
   CSVs + raw `.txt` files into
   `quant_sentiment.event_frame_v2.FilingEvent` records.
2. Load issuer + SPY prices from `yf_sentiment_equities_daily_2015_2025_v1`.
3. `quant_sentiment.event_frame_v2.build_event_frame(...)` to construct
   the event-level sample frame.
4. `quant_sentiment.historical_text_study_v2.run_period_study(...)` for
   `formation_dev` + `validation` (both `allow_holdout=False`), then once
   more for `historical_evaluation` with `allow_holdout=True` (the config
   is already `status: frozen-for-holdout`, so the gate passes) — labeled
   `PREVIOUSLY INSPECTED / HISTORICAL EVALUATION`, never "untouched
   holdout," per the config's own note (AAPL/MSFT/AMZN 2025 was already
   inspected under v1).
5. Write `research/historical-textual-signal-study.md` from the real
   results — every table quoted from the committed JSON artifacts, nulls
   reported exactly, no target-search if Model 3 doesn't beat Model 1.
6. Commit datasets + results + report, push, update PR, post a tracker
   checkpoint.

No runner script exists yet for step 4 (unlike D9-A/B/C's
`scripts/run_historical_*_study*.py`) — it was not written in this session
because there is no real data yet to run it against; writing one now would
be untested against the actual acquired file layout. Write it once the
real data's exact directory/file structure is confirmed from the handoff
run's own output.
