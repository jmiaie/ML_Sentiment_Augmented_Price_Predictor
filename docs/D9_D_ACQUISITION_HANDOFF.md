# D9-D Acquisition Handoff — 12-issuer SEC EDGAR filings + prices

**Status update (2026-09-17, later same day)**: the acquisition described
below is now running on a real host with genuine network access (Hai's
environment) — `data.sec.gov`, `www.sec.gov`, and `query1.finance.yahoo.com`
all measured reachable (200s, ≤0.25s) from there. A `--max-filings-per-symbol
1 --no-freeze` dry run succeeded: SEC leg 12/12 issuers ok, Yahoo leg 12
issuers + SPY at 2,766 rows each with zero missing. The two findings below
were caught and reported from that real run; both are fixed here.

**Original status (2026-09-17)**: all D9-D AUTHORITATIVE (v2) code and tests
are complete and passing on synthetic fixtures. The real 12-issuer SEC EDGAR
10-K/10-Q/8-K corpus had **not** been acquired in this session — this was
the one remaining step before `formation_dev`/`validation`/
`historical_evaluation` could run on real data and the required report
could be written.

## Why this was a handoff from this session, not a run here

This session's own egress proxy blocks both SEC hosts:

```
$ curl -sS --max-time 10 https://data.sec.gov/submissions/CIK0000320193.json
curl: (56) CONNECT tunnel failed, response 403
```

Confirmed for both `data.sec.gov` and `www.sec.gov` — **this session's own
egress proxy policy**, not a statement about SEC EDGAR's actual
availability (which a real host, per the status update above, reaches
without issue). Yahoo Finance access was not separately re-tested in this
session but the same acquisition script's price leg worked previously for
the v1 dataset in a prior working session, and is now independently
confirmed reachable too.

## What to run

From an environment with real network access to SEC EDGAR and Yahoo
Finance, with this repository checked out at this branch:

```bash
pip install -e ".[dev,data]"   # dev: pandas-market-calendars, pysentiment2, mypy/pytest/ruff.
                                # data: yfinance -- NOT in [dev]; omitting it makes
                                # download_prices() fail on import on a clean install.
python scripts/acquire_sec_filings_12issuer_daily.py
```

This acquires and freezes two datasets:

- `data/manifests/sec_filings_12issuer_2015_2025_v1.json` +
  `data/raw/sec_filings_12issuer_2015_2025_v1/` (gitignored, do not commit)
- `data/manifests/yf_sentiment_equities_daily_2015_2025_v1.json` +
  `data/raw/yf_sentiment_equities_daily_2015_2025_v1/` (gitignored)

Flags: `--no-freeze` for a validate-only dry run, `--max-filings-per-symbol N`
to sanity-check on a small slice first, `--raw-dir PATH` to redirect output.

## CIK map — now independently verified live, all 12 correct

`scripts/acquire_sec_filings_12issuer_daily.py`'s `SYMBOL_CIK` maps all 12
issuers to their SEC Central Index Keys. AAPL/MSFT/AMZN matched this
repo's already-frozen v1 acquisition exactly from the start (cross-checked
against `data/manifests/edgar_8k_yf_megacap_daily_2015_2025_v1.json`); the
other nine (GOOGL, NVDA, JPM, XOM, JNJ, PG, WMT, HD, KO) were flagged in
the original version of this doc as unverified, since no network was
available in the session that wrote them.

**Since verified live against `data.sec.gov/submissions/CIK##########.json`
(zero-padded to 10 digits) by Hai, from a host with real access**: all 12
match. 11 match on both `name` and `tickers`. XOM's CIK (`0000034088`) is
correct — SEC's own `name` field returns `EXXON MOBIL CORP`, confirming
it — but its `tickers` array is empty in that endpoint's response, which
is a known SEC data quirk for some large/older filers, not a wrong
mapping. No CIK in the map points at the wrong company.

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
be untested against the actual acquired file layout. **Hai has taken this
step**: writing it once the real acquisition trees exist, against their
actual layout, rather than a guessed one.

## Live status

As of this update, Hai is running the acquisition in two passes on their
own host: pass 1 (`--no-freeze`, full universe, no `--max-filings-per-symbol`
cap) validates the complete pull; pass 2 freezes once pass 1 reports zero
failures. Nothing has been merged or committed to this branch by that run.
Filing counts and the frozen manifests will follow from that side once
pass 1 completes.
