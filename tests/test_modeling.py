import json
from pathlib import Path

from quant_sentiment.features import (
    MARKET_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    build_modeling_frame,
)
from quant_sentiment.modeling import run_ablation_study
from quant_sentiment.synthetic_validation import (
    SYNTHETIC_LABEL,
    generate_synthetic_inputs,
    run_synthetic_methodology_validation,
)


def test_ablation_framework_detects_injected_incremental_signal() -> None:
    market_frame, events_frame = generate_synthetic_inputs()
    modeling_frame = build_modeling_frame(market_frame, events_frame)

    results = run_ablation_study(
        frame=modeling_frame,
        market_feature_columns=MARKET_FEATURE_COLUMNS,
        sentiment_feature_columns=SENTIMENT_FEATURE_COLUMNS,
    )

    combined = results["test_metrics"]["combined_logistic"]
    market_only = results["test_metrics"]["market_only_logistic"]
    sentiment_only = results["test_metrics"]["sentiment_only_logistic"]

    assert combined["samples"] == market_only["samples"] == sentiment_only["samples"]
    assert combined["log_loss"] < market_only["log_loss"]
    assert combined["log_loss"] < sentiment_only["log_loss"]
    assert combined["calibration_table"]


def test_synthetic_validation_writes_labeled_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "synthetic_methodology_validation.json"

    payload = run_synthetic_methodology_validation(output_path)
    stored = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["label"] == SYNTHETIC_LABEL
    assert stored["label"] == SYNTHETIC_LABEL
    assert "Historical results pending reproducible point-in-time dataset." in stored[
        "dataset_disclosure"
    ]
