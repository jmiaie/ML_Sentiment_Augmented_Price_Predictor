import numpy as np
import pandas as pd
import pytest

from quant_sentiment.walk_forward_v2 import build_purged_event_walk_forward_plan


def _irregular_multi_issuer_sessions(n: int, seed: int) -> tuple[pd.Series, pd.Series]:
    """Simulates a pooled, session-ordered, multi-issuer event frame:
    consecutive rows can share a session (same-day filings from different
    issuers) or be many sessions apart -- exactly the case a pure
    row-count gap gets wrong."""
    rng = np.random.default_rng(seed)
    gaps = rng.integers(0, 3, n)  # 0 = same session as the previous row
    effective = np.cumsum(gaps)
    label_horizon = rng.integers(1, 6, n)  # primary=1..secondary=5-ish
    target_end = effective + label_horizon
    return pd.Series(effective), pd.Series(target_end)


def test_embargo_is_respected_across_every_validation_fold() -> None:
    effective, target_end = _irregular_multi_issuer_sessions(200, seed=1)
    embargo = 5

    plan = build_purged_event_walk_forward_plan(
        effective,
        target_end,
        initial_train_size=40,
        validation_size=15,
        test_size=15,
        step_size=15,
        embargo_sessions=embargo,
    )

    assert len(plan.validation_splits) > 0
    for split in plan.validation_splits:
        eval_start_session = effective.iloc[split.evaluation_indices[0]]
        if len(split.train_indices) == 0:
            continue
        max_train_target_end = target_end.iloc[split.train_indices].max()
        assert eval_start_session - max_train_target_end >= embargo


def test_embargo_is_respected_on_the_final_test_split() -> None:
    effective, target_end = _irregular_multi_issuer_sessions(200, seed=2)
    embargo = 5

    plan = build_purged_event_walk_forward_plan(
        effective,
        target_end,
        initial_train_size=40,
        validation_size=15,
        test_size=15,
        step_size=15,
        embargo_sessions=embargo,
    )

    eval_start_session = effective.iloc[plan.test_indices[0]]
    max_train_target_end = target_end.iloc[plan.pre_test_train_indices].max()
    assert eval_start_session - max_train_target_end >= embargo


def test_larger_embargo_never_admits_more_training_rows() -> None:
    effective, target_end = _irregular_multi_issuer_sessions(200, seed=3)
    plan_small = build_purged_event_walk_forward_plan(
        effective,
        target_end,
        initial_train_size=40,
        validation_size=15,
        test_size=15,
        step_size=15,
        embargo_sessions=1,
    )
    plan_large = build_purged_event_walk_forward_plan(
        effective,
        target_end,
        initial_train_size=40,
        validation_size=15,
        test_size=15,
        step_size=15,
        embargo_sessions=10,
    )
    assert len(plan_large.pre_test_train_indices) <= len(plan_small.pre_test_train_indices)


def test_unsorted_effective_sessions_are_rejected() -> None:
    effective = pd.Series([5, 3, 8, 1])
    target_end = pd.Series([6, 4, 9, 2])
    with pytest.raises(ValueError):
        build_purged_event_walk_forward_plan(
            effective,
            target_end,
            initial_train_size=1,
            validation_size=1,
            test_size=1,
            step_size=1,
            embargo_sessions=0,
        )


def test_negative_embargo_is_rejected() -> None:
    effective = pd.Series(range(50))
    target_end = pd.Series(range(1, 51))
    with pytest.raises(ValueError):
        build_purged_event_walk_forward_plan(
            effective,
            target_end,
            initial_train_size=10,
            validation_size=5,
            test_size=5,
            step_size=5,
            embargo_sessions=-1,
        )
