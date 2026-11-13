"""Checks for Member 2's baselines on a tiny made-up dataset. Run with: pytest"""
import pandas as pd
import pytest

from hm_recsys.baselines import (
    AgeBandPopularityRecommender,
    BuyAgainRecommender,
    NewCustomerRecommender,
    PopularityRecommender,
    RandomRecommender,
    age_band,
    popularity_bias_summary,
)


@pytest.fixture
def data():
    # week 1 (old): item 1 is the best seller
    # week 3 (recent): young buy 10, older buy 20
    rows = [
        ("2020-09-01", "a", 1), ("2020-09-01", "b", 1), ("2020-09-01", "c", 1),
        ("2020-09-01", "d", 1), ("2020-09-02", "a", 2),
        ("2020-09-15", "young1", 10), ("2020-09-15", "young2", 10), ("2020-09-16", "young1", 11),
        ("2020-09-15", "old1", 20), ("2020-09-16", "old2", 20), ("2020-09-16", "old3", 20),
        ("2020-09-16", "a", 3), ("2020-09-16", "a", 3), ("2020-09-16", "a", 3),
    ]
    train = pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"])
    train["t_dat"] = pd.to_datetime(train["t_dat"])
    customers = pd.DataFrame({
        "customer_id": ["a", "b", "c", "d", "young1", "young2", "old1", "old2", "old3",
                        "new_young", "new_noage"],
        "age": [30, 30, 30, 30, 20, 21, 60, 61, 62, 19, None],
    })
    articles = pd.DataFrame({"article_id": [1, 2, 3, 10, 11, 20, 30, 40, 50, 60, 70, 80, 90, 99]})
    return train, customers, articles


ALL_MODELS = [
    lambda: RandomRecommender(),
    lambda: PopularityRecommender(),
    lambda: PopularityRecommender(weeks=1),
    lambda: AgeBandPopularityRecommender(weeks=1),
    lambda: NewCustomerRecommender(weeks=1),
    lambda: BuyAgainRecommender(),
]


@pytest.mark.parametrize("make", ALL_MODELS)
@pytest.mark.parametrize("cid", ["a", "young1", "new_young", "new_noage", "never-seen-id"])
def test_every_model_returns_k_unique_items_for_any_customer(data, make, cid):
    train, customers, articles = data
    model = make().fit(train, articles, customers)
    recs = model.recommend(cid, k=5)
    assert len(recs) == 5
    assert len(set(recs)) == 5
    assert all(isinstance(i, int) for i in recs)
    assert model.recommend_all([cid], k=5)[cid] == recs


def test_random_is_reproducible_and_differs_between_customers(data):
    train, customers, articles = data
    m = RandomRecommender(seed=1).fit(train, articles)
    assert m.recommend("x", 5) == m.recommend("x", 5)
    assert m.recommend("x", 5) != m.recommend("y", 5)


def test_global_popularity_counts_distinct_buyers(data):
    train, *_ = data
    # item 3 has three rows but one buyer, so it must rank below item 1 (four buyers)
    top = PopularityRecommender().fit(train).recommend("x", 10)
    assert top[0] == 1
    assert top.index(20) < top.index(3)
    assert PopularityRecommender(count_by="rows").fit(train).recommend("x", 2) == [1, 3]


def test_trending_uses_only_recent_weeks(data):
    train, *_ = data
    top = PopularityRecommender(weeks=1).fit(train).recommend("x", 2)
    assert top == [20, 10]
    assert 1 not in PopularityRecommender(weeks=1).fit(train).recent_


def test_age_band_path(data):
    train, customers, articles = data
    m = AgeBandPopularityRecommender(weeks=1).fit(train, articles, customers)
    assert m.recommend("new_young", 1) == [10]        # young trend
    assert m.recommend("old1", 1) == [20]             # older trend
    assert m.recommend("new_noage", 1) == [20]        # no age -> global trend
    assert m.recommend("never-seen-id", 1) == [20]    # unknown id -> global trend


def test_buy_again_puts_own_items_first_then_trends(data):
    train, customers, articles = data
    m = BuyAgainRecommender().fit(train, articles, customers)
    recs = m.recommend("a", 5)
    assert recs[0] == 3                  # bought three times
    assert set(recs[:3]) == {1, 2, 3}
    assert m.recommend("new_young", 1) == [10]   # no history -> new-customer path


def test_age_band_labels():
    bands = age_band(pd.Series([16, 24.9, 25, 64, 65, None]))
    assert bands.tolist() == ["16-24", "16-24", "25-34", "55-64", "65+", "unknown"]


def test_popularity_bias_summary(data):
    train, *_ = data
    preds = PopularityRecommender().fit(train).recommend_all(["x", "y"], k=2)
    s = popularity_bias_summary(preds, train)
    assert s["distinct_items_shown"] == 2
    assert 0 < s["share_sold_items_never_shown"] < 1
