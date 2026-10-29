import pytest

from src.gcn.pruning_reporting import condition_name, method_series, random_seed_table


def _condition(test_acc, actual_fraction):
    return {"test_acc": test_acc, "actual_fraction": actual_fraction, "test_loss_pp": 100 * (0.8 - test_acc),
            "disagree_test": 1 - test_acc}


@pytest.fixture
def conditions():
    return {
        ("none", None, 0.0): _condition(0.80, 0.0),
        ("random", 101, 0.5): _condition(0.70, 0.4999),
        ("random", 102, 0.5): _condition(0.74, 0.4999),
        ("random", 103, 0.5): _condition(0.72, 0.4999),
        ("similarity", None, 0.5): _condition(0.60, 0.4999),
    }


def test_method_series_starts_at_the_shared_baseline_and_uses_actual_fraction(conditions):
    series = method_series(conditions, lambda c: 100 * c["test_acc"])
    assert list(series) == ["random", "similarity"]
    for points in series.values():
        assert points[0] == (0.0, 80.0, 80.0, 80.0)
        assert points[1][0] == pytest.approx(49.99)
    # Random: mean over pruning seeds with the seed range.
    assert series["random"][1][1:] == pytest.approx((72.0, 70.0, 74.0))
    assert series["similarity"][1][1:] == pytest.approx((60.0, 60.0, 60.0))


def test_random_seed_table_summarizes_three_seeds(conditions):
    headers, rows = random_seed_table(conditions)
    assert len(rows) == 1
    assert rows[0][1] == 3
    assert rows[0][2] == "72.0% [70.0%–74.0%]"
    assert rows[0][3] == "+8.0 [+6.0–+10.0]"


def test_condition_names():
    assert condition_name("none", None, 0.0) == "original graph"
    assert condition_name("random", 101, 0.25) == "random s101 25%"
    assert condition_name("similarity", None, 0.75) == "feature similarity 75%"
