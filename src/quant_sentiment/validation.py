from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WalkForwardSplit:
    name: str
    train_indices: np.ndarray
    evaluation_indices: np.ndarray


@dataclass(frozen=True)
class WalkForwardPlan:
    validation_splits: list[WalkForwardSplit]
    pre_test_train_indices: np.ndarray
    test_indices: np.ndarray
    gap: int


def build_walk_forward_plan(
    n_samples: int,
    initial_train_size: int,
    validation_size: int,
    test_size: int,
    step_size: int,
    label_horizon: int = 1,
    embargo: int = 0,
) -> WalkForwardPlan:
    if min(n_samples, initial_train_size, validation_size, test_size, step_size) <= 0:
        raise ValueError("Split sizes must all be positive.")
    if label_horizon <= 0 or embargo < 0:
        raise ValueError("label_horizon must be positive and embargo must be non-negative.")

    gap = max(label_horizon - 1, 0) + embargo
    development_observations = n_samples - test_size - gap
    if development_observations <= initial_train_size + validation_size:
        raise ValueError("Not enough observations for validation folds and a final test window.")

    validation_splits: list[WalkForwardSplit] = []
    train_end = initial_train_size
    fold_number = 1

    while train_end + gap + validation_size <= development_observations:
        evaluation_start = train_end + gap
        evaluation_end = evaluation_start + validation_size
        validation_splits.append(
            WalkForwardSplit(
                name=f"validation_fold_{fold_number}",
                train_indices=np.arange(0, train_end, dtype=int),
                evaluation_indices=np.arange(evaluation_start, evaluation_end, dtype=int),
            )
        )
        train_end += step_size
        fold_number += 1

    if not validation_splits:
        raise ValueError("Unable to construct any walk-forward validation splits.")

    pre_test_train_indices = np.arange(0, development_observations, dtype=int)
    test_start = development_observations + gap
    test_indices = np.arange(test_start, test_start + test_size, dtype=int)

    return WalkForwardPlan(
        validation_splits=validation_splits,
        pre_test_train_indices=pre_test_train_indices,
        test_indices=test_indices,
        gap=gap,
    )


def labels_are_non_overlapping(split: WalkForwardSplit, label_horizon: int) -> bool:
    return bool(split.evaluation_indices[0] - split.train_indices[-1] >= label_horizon)
