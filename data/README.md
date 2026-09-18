# Data layout (Directive #9)

| Path | Tracked? | Purpose |
|---|---|---|
| `data/raw/` | **No** (gitignored) | Frozen EDGAR text + yfinance price snapshots. Never commit. |
| `data/manifests/` | **Yes** | Provenance manifests + SHA-256 after freeze. |

**Acquisition:** local/agent only via `scripts/acquire_edgar_8k_yf_megacap_daily.py`. Never from CI.

**Scope:** megacap 8-K subset (AAPL/MSFT/AMZN) + daily prices. Not full EDGAR corpus.
