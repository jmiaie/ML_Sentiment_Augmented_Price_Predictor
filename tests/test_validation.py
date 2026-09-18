import pandas as pd

from quant_sentiment.validation import build_walk_forward_plan, labels_are_non_overlapping


def test_walk_forward_plan_applies_horizon_gap_and_embargo() -> None:
    plan = build_walk_forward_plan(
        n_samples=80,
        initial_train_size=20,
        validation_size=10,
        test_size=10,
        step_size=10,
        label_horizon=2,
        embargo=1,
    )

    assert plan.gap == 3  # label_horizon + embargo, not (label_horizon - 1) + embargo
    for split in plan.validation_splits:
        assert labels_are_non_overlapping(split, label_horizon=2)
        assert split.evaluation_indices[0] - split.train_indices[-1] >= 2

    assert plan.test_indices[0] - plan.pre_test_train_indices[-1] >= 2


def test_gap_formula_prevents_the_exact_label_horizon_1_embargo_0_leak() -> None:
    """Regression test for a real defect: with the frozen D9-D config's own
    label_horizon=1, embargo=0, the old formula `gap = max(label_horizon-1,
    0) + embargo` produced gap=0, and the last training row's own label was
    numerically IDENTICAL to the first evaluation row's own feature (both
    reduce to close[t+1]/close[t] - 1 for the same t) -- a real leak that
    the old `labels_are_non_overlapping` check (`>= label_horizon`) failed
    to catch, since 1 >= 1 is True even though an exact overlap exists.
    Verified here end-to-end with the actual label/feature formulas, not
    just the index arithmetic."""
    plan = build_walk_forward_plan(
        n_samples=80,
        initial_train_size=20,
        validation_size=10,
        test_size=10,
        step_size=10,
        label_horizon=1,
        embargo=0,
    )

    assert plan.gap == 1  # NOT 0 -- the old formula's max(1-1,0)+0 result
    split = plan.validation_splits[0]
    assert labels_are_non_overlapping(split, label_horizon=1)

    close = pd.Series(range(1, 100)).astype(float)  # arbitrary monotonic series
    next_session_return = close.shift(-1).div(close).sub(1.0)  # labels.py's formula
    market_return_1d = close.pct_change()  # features.py's formula

    last_train_idx = split.train_indices[-1]
    first_eval_idx = split.evaluation_indices[0]
    last_train_label = next_session_return.iloc[last_train_idx]
    first_eval_feature = market_return_1d.iloc[first_eval_idx]

    # With the fix, these must NOT be the same value -- under the old
    # gap=0 formula they were identical for every fold, every period.
    assert last_train_label != first_eval_feature
