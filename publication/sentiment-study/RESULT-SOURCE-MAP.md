# RESULT-SOURCE-MAP — D10-D (sentiment-study)

Every table, figure and numeric claim in this pack resolves to an accepted artifact
under `results/`, the append-only ledger, or the frozen configuration. No number in
this pack originates outside those files.

Citation rows are machine-verified: `publication_pack.py check` re-hashes every path
below and fails closed on a mismatch. Paths are repository-relative; hashes are
SHA-256 of the exact file bytes at the accepted HEAD
`9184eff7571f8911f410632df8a06601c98311bc`.

## Primary citation table

| id | claim it supports | path | sha256 |
| --- | --- | --- | --- |
| C-01 | 2025 headline comparison (all §7 numbers, primary target) | results/historical_text_v2/sentiment_hist_text_v2_primary_historical_evaluation.json | 6ed9197eb2d51cd68341a7fc1a9790eb61d6fa9ad10fda9e148a8da5120cca36 |
| C-02 | 2025 secondary target comparison (§7 robustness) | results/historical_text_v2/sentiment_hist_text_v2_secondary_historical_evaluation.json | b207ed7a5c7923f6bdb0217bdc37963b5e81b37a2dc34788b0696ef822557d12 |
| C-03 | 2024 validation block, primary target (§8) | results/historical_text_v2/sentiment_hist_text_v2_primary_validation.json | 5eb3d5b839d49672398c3e5bc681f47fe79339ea9fb7f819746a327fdcf7f2a3 |
| C-04 | 2024 validation block, secondary target (§8) | results/historical_text_v2/sentiment_hist_text_v2_secondary_validation.json | e2e574e959fe5d8f80ab2e4fc072de49ff92164a28a6929ffebdd83f39330e40 |
| C-05 | development internal test block, primary target (§8) | results/historical_text_v2/sentiment_hist_text_v2_primary_formation_dev.json | 3ccbcf530f8c09f7860b261b8c0c136d4a3a233f142388f29a3138ce438cd359 |
| C-06 | development internal test block, secondary target (§8) | results/historical_text_v2/sentiment_hist_text_v2_secondary_formation_dev.json | a8ce502eabf3d39d0ff665da46d8debcdb0a8e14a862124ee77180d9017a79f0 |
| C-07 | provenance of the frozen C values (§4, §6) | results/historical_text_v2/sentiment_hist_text_v2_final_c_selection.json | 725f8787b71a866a483abf8f567c09216e9af942c2aa3fa6672d03c026adab6f |
| C-08 | universe, features, targets, C grid, periods (§2–§4) | configs/experiments/sentiment_historical_text_study_v2.yaml | bd702bf5f58311b80ad68dc0fc9bb89bbcd11479ddd49c480e7fd6beacdd80f8 |
| C-09 | filings corpus composition, per-file hashes, canonical dataset hash (§2) | data/manifests/sec_filings_12issuer_2015_2025_v1.json | 9ad504c9335fb7fafc2f1722bdcd33347bcb034c02013f1fafbb329e3ad6a6cc |
| C-10 | prices corpus, 13 per-file hashes, canonical dataset hash (§2) | data/manifests/yf_sentiment_equities_daily_2015_2025_v1.json | 32c0f47c61c67601775ff3be3747b6ec4e62626fc6e805a1dc1731498a4bacc4 |
| C-11 | append-only experiment record; artifact hashes per run (§6, §11) | research/experiment-ledger.csv | 19df8052ddfcb548dd45c630d79e3c206e76067effa0eb168d94335295c82437 |
| C-12 | classification rules and the pre-committed 2025 label (§1, §9) | research/holdout-audit.md | 8a80bdcf18de87ef196bbd5048d90879de95f6f8d5cf83f65f61f6eddc2939eb |
| C-13 | fold formation, C selection, target-aware period slicing (§4–§6) | src/quant_sentiment/historical_text_study_v2.py | e1b0895df8eabb11d1f2768cc21e6735ed53ba28d17e18438e6fcf588ca4e446 |
| C-14 | fail-closed frozen-input verification before any fitting (§2, §5) | src/quant_sentiment/frozen_inputs.py | 2a1fc3b93e51df74e88344e5abd5e152daee126a65438285ef9ea8671d49d7ff |
| C-15 | the single 2025 invocation and its CLI contract (§11) | scripts/run_historical_text_study_v2.py | 66688835ae5b3beda127276a0368ae9182cdb465ff40e99f4222e530d57d057b |
| C-16 | superseded v1 2025 run — preserved, not a publication source (§9) | results/historical_text/sentiment_hist_text_v1_holdout_2025.json | 216896f0b46fd8854ecb1f042816fcdca29d9fabcf65fbdae81a12ffb87365d9 |
| C-17 | superseded v1 2024 run (§9) | results/historical_text/sentiment_hist_text_v1_val_2024.json | 4d15f062c6c048b7335d2ddd20eb2dfe114c7e2ce50629409843b297adc54e4a |
| C-18 | superseded v1 development run (§9) | results/historical_text/sentiment_hist_text_v1_dev_formation.json | 04aaf601ba72e38153815247532d4005556f377eb5c7b4fe5e09db26e037ac87 |

## Generated (non-source) artifacts

These are produced deterministically by `publication_pack.py build` and verified
byte-for-byte on every run. They are presented as convenience views; every value in
them is copied straight out of the artifacts above, and each is re-derived by
`claim_crosscheck.py` from the artifacts rather than from the generated file.

| id | claim it supports | path | sha256 |
| --- | --- | --- | --- |
| G-01 | per-period × target × model metrics as recorded (§7, §8) | publication/sentiment-study/tables/period-metrics.csv | c2b8f24f310458b1ab3e043c0b3c5445b80e69bcdb5a07872b4846b01e15e473 |
| G-02 | per-period × target headline comparison and geometry (§7) | publication/sentiment-study/tables/target-comparison.csv | ab8a05eaab6f0e8f8c72d2cccf09a96db3eae45d010104331dbf228ebd6c3138 |
| G-03 | measured vs declared hash for every source artifact (§11) | publication/sentiment-study/tables/artifact-hashes.csv | 30e38dca9f16da3306776941999b88c07cc0da68fd9aa97c6f0933d47b972bfa |
| G-04 | 2025 log loss by model and target, visual form (§7) | publication/sentiment-study/figures/log-loss-comparison.svg | 201e24528813dbf59bbc943a72fb2910afa25f6b6d89748935646eb97b433c9a |

**Generated-file rows.** Each `G-0x` row carries the SHA-256 of the generated file
at the time this pack was built. That value is not taken on trust: `check` rebuilds
every generated file and compares bytes, re-hashes each row's declared value, and CI
additionally requires `git diff --exit-code` to be clean after `build`. A generated
file that stops being reproducible fails three independent ways.

## Numbers and where they come from

| quantity | source field | citation |
| --- | --- | --- |
| 2025 headline Δ log loss, primary | `result.key_metrics.headline_delta_log_loss_model3_minus_model1` | C-01 |
| 2025 bootstrap interval, primary | `result.key_metrics.headline_bootstrap_delta_log_loss_ci95` | C-01 |
| 2025 headline Δ log loss, secondary | same field, secondary target | C-02 |
| 2024 validation Δ log loss | same field | C-03, C-04 |
| development internal test Δ log loss | same field | C-05, C-06 |
| corpus composition, skips, frame rows | `corpus` | C-01 |
| frozen-input verification record | `input_integrity` | C-01 |
| training/evaluation geometry, C provenance | `result.key_metrics.pre_evaluation_training`, `c_selection` | C-01, C-02 |
| binding 2025 label | `historical_evaluation_label` | C-01, C-02, C-12 |

## Artifacts that do NOT exist and are therefore not cited

- `research/historical-textual-signal-study.md` — named as the required report by
  the frozen configuration (C-08) but absent at this HEAD. This pack is the first
  document to attempt that report; the absence is disclosed as a limitation, not
  papered over. No prose finding in this pack is sourced from it.
- Any 2025-period result artifact beyond C-01 and C-02. Exactly two 2025 artifacts
  exist. The ledger (C-11) records one 2025 invocation, for two targets.
