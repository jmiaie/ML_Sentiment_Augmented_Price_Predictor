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

    assert plan.gap == 2
    for split in plan.validation_splits:
        assert labels_are_non_overlapping(split, label_horizon=2)
        assert split.evaluation_indices[0] - split.train_indices[-1] >= 2

    assert plan.test_indices[0] - plan.pre_test_train_indices[-1] >= 2
