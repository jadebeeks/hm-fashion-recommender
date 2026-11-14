"""Quick checks for the metrics. Owner: Member 1. Run with: pytest"""
import pytest
import pandas as pd
from hm_recsys.base import Recommender
from hm_recsys.evaluation import evaluate

from hm_recsys.evaluation import (
    average_precision_at_k,
    map_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    catalog_coverage,
    gini_index,
    intra_list_diversity,
    novelty,
)

def test_perfect_prediction_scores_one():
    assert average_precision_at_k({1, 2}, [1, 2, 3], k=12) == pytest.approx(1.0)

def test_no_hits_scores_zero():
    assert average_precision_at_k({9}, [1, 2, 3], k=12) == 0.0


def test_map_at_k():
    actual = {"C1": {1, 2}, "C2": {5}, "C3": set()}
    predicted = {"C1": [1, 2], "C2": [7, 5], "C3": [9]}

    assert map_at_k(actual, predicted) == pytest.approx(0.75)


def test_recall_at_k():
    actual = {"C1": {1, 2, 3}, "C2": {5, 6}}
    predicted = {"C1": [3, 1], "C2": [5]}

    assert recall_at_k(actual, predicted) == pytest.approx(7 / 12)


def test_ndcg_at_k():
    actual = {"C1": {1, 2}}

    perfect = {"C1": [1, 2]}
    imperfect = {"C1": [1, 3, 2]}

    assert ndcg_at_k(actual, perfect) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(actual, imperfect) < 1


def test_hit_rate_at_k():
    actual = {"C1": {1, 2}, "C2": {5}}
    predicted = {"C1": [2, 3], "C2": [7, 8]}

    assert hit_rate_at_k(actual, predicted) == pytest.approx(0.5)


def test_catalog_coverage():
    predicted = {"C1": [101, 102], "C2": [102, 103]}

    assert catalog_coverage(predicted, 10) == pytest.approx(0.3)
    assert catalog_coverage({}, 10) == 0.0


def test_gini_index():
    even = {"C1": [101, 102], "C2": [101, 102]}
    uneven = {"C1": [101, 101, 101], "C2": [102]}

    assert gini_index(even) == pytest.approx(0.0)
    assert gini_index(uneven) == pytest.approx(0.25)


def test_intra_list_diversity():
    articles = pd.DataFrame({
        "article_id": [101, 102, 103, 104],
        "product_group_name": ["Shirts", "Shirts", "Trousers", "Dresses"]
    })

    predicted = {"C1": [101, 102, 103], "C2": [101, 104]}

    assert intra_list_diversity(predicted, articles) == pytest.approx(5 / 6)


def test_novelty():
    popularity = pd.Series({101: 100, 102: 10, 103: 1})

    popular = {"C1": [101]}
    rare = {"C1": [103]}

    assert novelty(rare, popularity) > novelty(popular, popularity)
    assert novelty({}, popularity) == 0.0


def test_evaluate_integration():

    class DemoRecommender(Recommender):

        def fit(self, train, articles=None, customers=None):
            return self

        def recommend(self, customer_id, k=12):
            recommendations = {
                "C1": [101, 105],
                "C2": [103, 105],
                "C3": [105, 104],
            }
            return recommendations[customer_id][:k]

    train = pd.DataFrame({
        "customer_id": ["C1"] + ["C2"] * 5,
        "article_id": [101] + [103] * 5
    })

    test = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "article_id": [101, 103, 104]
    })

    articles = pd.DataFrame({
        "article_id": [101, 102, 103, 104, 105],
        "product_group_name": [
            "Tops", "Tops", "Bottoms", "Shoes", "Accessories"
        ]
    })

    customers = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "age": [22, 37, None]
    })

    model = DemoRecommender().fit(train)

    results = evaluate(
        model,
        test,
        k=2,
        train=train,
        articles=articles,
        customers=customers
    )

    assert results.loc["all", "MAP@2"] == pytest.approx(5 / 6)
    assert results.loc["all", "Recall@2"] == pytest.approx(1.0)
    assert results.loc["all", "HitRate@2"] == pytest.approx(1.0)
    assert results.loc["all", "coverage"] == pytest.approx(0.8)
    assert results.loc["all", "gini"] == pytest.approx(0.25)
    assert results.loc["all", "diversity"] == pytest.approx(1.0)

    assert results.loc["new", "n_customers"] == 1
    assert results.loc["light", "n_customers"] == 1
    assert results.loc["regular", "n_customers"] == 1

    assert results.loc["age:0-24", "n_customers"] == 1
    assert results.loc["age:35-44", "n_customers"] == 1
    assert results.loc["age:unknown", "n_customers"] == 1
