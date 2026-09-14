from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .validation import WalkForwardPlan, WalkForwardSplit, build_walk_forward_plan


@dataclass(frozen=True)
class ModelSelectionResult:
    best_c: float | None
    validation_metrics: list[dict[str, Any]]


def _build_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=c_value, max_iter=1_000, random_state=7)),
        ]
    )


def _calibration_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 5,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if not np.any(mask):
            continue
        records.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": int(mask.sum()),
                "mean_predicted_probability": float(probabilities[mask].mean()),
                "observed_positive_rate": float(y_true[mask].mean()),
            }
        )
    return records


def _classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    metrics: dict[str, Any] = {
        "samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "calibration_table": _calibration_table(y_true, probabilities),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities)) if np.unique(y_true).size > 1 else None
    )
    return metrics


def _fit_probabilities(
    train_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    c_value: float | None,
) -> np.ndarray:
    y_train = train_frame[target_column].astype(int).to_numpy()
    if np.unique(y_train).size < 2 or c_value is None:
        return np.full(evaluation_frame.shape[0], float(y_train.mean()) if y_train.size else 0.5)

    model = _build_pipeline(c_value)
    model.fit(train_frame[feature_columns], y_train)
    return model.predict_proba(evaluation_frame[feature_columns])[:, 1]


def _evaluate_fold(
    frame: pd.DataFrame,
    split: WalkForwardSplit,
    feature_columns: list[str],
    target_column: str,
    c_value: float | None,
) -> dict[str, Any]:
    train_frame = frame.iloc[split.train_indices]
    evaluation_frame = frame.iloc[split.evaluation_indices]
    probabilities = _fit_probabilities(
        train_frame=train_frame,
        evaluation_frame=evaluation_frame,
        feature_columns=feature_columns,
        target_column=target_column,
        c_value=c_value,
    )
    metrics = _classification_metrics(
        evaluation_frame[target_column].astype(int).to_numpy(),
        probabilities,
    )
    metrics["fold"] = split.name
    return metrics


def _tune_logistic_model(
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    feature_columns: list[str],
    target_column: str,
    c_values: tuple[float, ...],
) -> ModelSelectionResult:
    best_c: float | None = None
    best_loss = inf
    best_metrics: list[dict[str, Any]] = []

    for c_value in c_values:
        fold_metrics = [
            _evaluate_fold(
                frame=frame,
                split=split,
                feature_columns=feature_columns,
                target_column=target_column,
                c_value=c_value,
            )
            for split in splits
        ]
        mean_log_loss = sum(metric["log_loss"] for metric in fold_metrics) / len(fold_metrics)
        if mean_log_loss < best_loss:
            best_loss = mean_log_loss
            best_c = c_value
            best_metrics = fold_metrics

    return ModelSelectionResult(best_c=best_c, validation_metrics=best_metrics)


def _majority_validation_metrics(
    frame: pd.DataFrame,
    splits: list[WalkForwardSplit],
    target_column: str,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for split in splits:
        train_frame = frame.iloc[split.train_indices]
        evaluation_frame = frame.iloc[split.evaluation_indices]
        probability = float(train_frame[target_column].astype(int).mean())
        probabilities = np.full(evaluation_frame.shape[0], probability)
        fold_metrics = _classification_metrics(
            evaluation_frame[target_column].astype(int).to_numpy(),
            probabilities,
        )
        fold_metrics["fold"] = split.name
        metrics.append(fold_metrics)
    return metrics


def run_ablation_study(
    frame: pd.DataFrame,
    market_feature_columns: list[str],
    sentiment_feature_columns: list[str],
    target_column: str = "next_session_direction",
    initial_train_size: int = 60,
    validation_size: int = 20,
    test_size: int = 20,
    step_size: int = 20,
    label_horizon: int = 1,
    embargo: int = 0,
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0),
) -> dict[str, Any]:
    working = frame.dropna(subset=[target_column]).reset_index(drop=True)
    if working.shape[0] <= initial_train_size + validation_size + test_size:
        raise ValueError("Not enough observations for the requested walk-forward plan.")

    plan: WalkForwardPlan = build_walk_forward_plan(
        n_samples=working.shape[0],
        initial_train_size=initial_train_size,
        validation_size=validation_size,
        test_size=test_size,
        step_size=step_size,
        label_horizon=label_horizon,
        embargo=embargo,
    )

    model_specs = {
        "majority_baseline": [],
        "market_only_logistic": market_feature_columns,
        "sentiment_only_logistic": sentiment_feature_columns,
        "combined_logistic": [*market_feature_columns, *sentiment_feature_columns],
    }

    selected_regularization: dict[str, float | None] = {}
    validation_metrics: dict[str, list[dict[str, Any]]] = {}

    for model_name, feature_columns in model_specs.items():
        if not feature_columns:
            selected_regularization[model_name] = None
            validation_metrics[model_name] = _majority_validation_metrics(
                frame=working,
                splits=plan.validation_splits,
                target_column=target_column,
            )
            continue

        selection_result = _tune_logistic_model(
            frame=working,
            splits=plan.validation_splits,
            feature_columns=feature_columns,
            target_column=target_column,
            c_values=c_values,
        )
        selected_regularization[model_name] = selection_result.best_c
        validation_metrics[model_name] = selection_result.validation_metrics

    pre_test_train_frame = working.iloc[plan.pre_test_train_indices]
    test_frame = working.iloc[plan.test_indices]
    test_metrics: dict[str, dict[str, Any]] = {}

    for model_name, feature_columns in model_specs.items():
        probabilities = _fit_probabilities(
            train_frame=pre_test_train_frame,
            evaluation_frame=test_frame,
            feature_columns=feature_columns,
            target_column=target_column,
            c_value=selected_regularization[model_name],
        )
        test_metrics[model_name] = _classification_metrics(
            test_frame[target_column].astype(int).to_numpy(),
            probabilities,
        )

    return {
        "gap": plan.gap,
        "label_horizon": label_horizon,
        "validation_split_count": len(plan.validation_splits),
        "selected_regularization": selected_regularization,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "test_window": {
            "start_session": str(test_frame["session"].iloc[0]),
            "end_session": str(test_frame["session"].iloc[-1]),
            "samples": int(test_frame.shape[0]),
        },
    }
