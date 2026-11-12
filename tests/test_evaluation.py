"""Quick checks for the metrics. Owner: Member 1. Run with: pytest"""
import pytest

from hm_recsys.evaluation import average_precision_at_k


@pytest.mark.skip(reason="enable once Member 1 implements average_precision_at_k")
def test_perfect_prediction_scores_one():
    assert average_precision_at_k({1, 2}, [1, 2, 3], k=12) == pytest.approx(1.0)


@pytest.mark.skip(reason="enable once Member 1 implements average_precision_at_k")
def test_no_hits_scores_zero():
    assert average_precision_at_k({9}, [1, 2, 3], k=12) == 0.0
