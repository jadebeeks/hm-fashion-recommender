import pandas as pd
import pytest
from scipy.sparse import isspmatrix_csr

from hm_recsys.baselines import PopularityRecommender
from hm_recsys.collaborative import ALSRecommender, ItemItemCF


@pytest.fixture
def train():
    rows = [
        ("u1", 1), ("u1", 2), ("u1", 2),
        ("u2", 1), ("u2", 2), ("u2", 3),
        ("u3", 1), ("u3", 3),
        ("u4", 4), ("u4", 5),
        ("u5", 4), ("u5", 6),
        ("u6", 4), ("u6", 6),
        ("u7", 4), ("u7", 5),
    ]
    return pd.DataFrame(rows, columns=["customer_id", "article_id"])


def test_fit_uses_sparse_binary_customer_article_interactions(train):
    model = ItemItemCF().fit(train)

    assert isspmatrix_csr(model.interaction_matrix_)
    assert isspmatrix_csr(model.similarity_matrix_)
    assert model.interaction_matrix_.shape == (7, 6)
    assert model.interaction_matrix_.nnz == 15
    assert set(model.interaction_matrix_.data) == {1.0}
    assert model.fit(train) is model


def test_recommendations_rank_similar_items_and_exclude_purchases(train):
    model = ItemItemCF().fit(train)

    recommendations = model.recommend("u1", k=4)

    assert recommendations[0] == 3
    assert not {1, 2}.intersection(recommendations)


def test_popularity_fallback_is_not_used_when_personalized_top_k_is_full(
    train, monkeypatch
):
    model = ItemItemCF().fit(train)

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("Popularity fallback should not be called.")

    monkeypatch.setattr(model.fallback_, "recommend", unexpected_fallback)

    assert model.recommend("u1", k=1) == [3]


def test_recommendations_are_unique_when_filled_from_popularity(train):
    model = ItemItemCF(n_neighbors=1).fit(train)

    recommendations = model.recommend("u1", k=4)

    assert len(recommendations) == 4
    assert len(set(recommendations)) == len(recommendations)
    assert not {1, 2}.intersection(recommendations)
    assert model.similarity_matrix_.getnnz(axis=1).max() <= 1


def test_unknown_customer_uses_popularity_fallback(train):
    model = ItemItemCF().fit(train)
    fallback = PopularityRecommender().fit(train)

    assert model.recommend("unknown", k=4) == fallback.recommend("unknown", k=4)


def test_als_fit_builds_sparse_binary_interactions(train):
    model = ALSRecommender(factors=4, iterations=2).fit(train)

    assert model.interaction_matrix_.shape == (7, 6)
    assert model.interaction_matrix_.nnz == 15
    assert set(model.interaction_matrix_.data) == {1.0}
    assert model.fit(train) is model


def test_als_recommendations_are_unique_and_exclude_purchases(train):
    model = ALSRecommender(factors=4, iterations=2).fit(train)

    recommendations = model.recommend("u1", k=4)

    assert len(recommendations) == 4
    assert len(set(recommendations)) == 4
    assert not {1, 2}.intersection(recommendations)


def test_als_uses_popularity_when_personalized_results_are_insufficient(
    train, monkeypatch
):
    model = ALSRecommender(factors=4, iterations=2).fit(train)
    monkeypatch.setattr(model.model_, "recommend", lambda *args, **kwargs: ([], []))

    recommendations = model.recommend("u1", k=4)

    assert len(recommendations) == 4
    assert len(set(recommendations)) == 4
    assert not {1, 2}.intersection(recommendations)


def test_als_fallback_searches_full_popularity_ranking(train, monkeypatch):
    model = ALSRecommender(factors=4, iterations=2).fit(train)
    monkeypatch.setattr(model.model_, "recommend", lambda *args, **kwargs: ([2], [1.0]))

    original_recommend = model.fallback_.recommend
    requested_counts = []

    def record_fallback_count(customer_id, k=12):
        requested_counts.append(k)
        return original_recommend(customer_id, k)

    monkeypatch.setattr(model.fallback_, "recommend", record_fallback_count)
    recommendations = model.recommend("u1", k=3)

    assert requested_counts == [len(model.fallback_.ranking_)]
    assert len(recommendations) == 3
    assert len(set(recommendations)) == 3
    assert not {1, 2}.intersection(recommendations)


def test_als_ignores_invalid_and_duplicate_item_indices(train, monkeypatch):
    model = ALSRecommender(factors=4, iterations=2).fit(train)
    invalid_indices = [-1, len(model.article_ids_), 2, 2]
    monkeypatch.setattr(
        model.model_,
        "recommend",
        lambda *args, **kwargs: (invalid_indices, [1.0] * len(invalid_indices)),
    )

    recommendations = model.recommend("u1", k=4)

    assert len(recommendations) == 4
    assert len(set(recommendations)) == 4
    assert set(recommendations).issubset(set(model.article_ids_))
    assert not {1, 2}.intersection(recommendations)


@pytest.mark.parametrize(
    "model_factory",
    [
        ItemItemCF,
        lambda: ALSRecommender(factors=2, iterations=1),
    ],
)
def test_models_return_fewer_than_k_when_only_one_article_is_eligible(model_factory):
    train = pd.DataFrame(
        [("u1", 101), ("u2", 202)],
        columns=["customer_id", "article_id"],
    )
    model = model_factory().fit(train)

    recommendations = model.recommend("u1", k=4)
    unknown_recommendations = model.recommend("unknown", k=4)

    assert recommendations == [202]
    assert len(set(recommendations)) == len(recommendations)
    assert len(unknown_recommendations) == 2
    assert len(set(unknown_recommendations)) == 2
    assert set(unknown_recommendations) == {101, 202}


@pytest.mark.parametrize(
    "model_factory",
    [
        ItemItemCF,
        lambda: ALSRecommender(factors=2, iterations=1),
    ],
)
def test_models_return_no_items_when_customer_bought_entire_catalog(model_factory):
    train = pd.DataFrame(
        [("u1", 101), ("u1", 202), ("u2", 101), ("u2", 202)],
        columns=["customer_id", "article_id"],
    )
    model = model_factory().fit(train)

    assert model.recommend("u1", k=4) == []


def test_als_unknown_customer_uses_popularity_fallback(train):
    model = ALSRecommender(factors=4, iterations=2).fit(train)
    fallback = PopularityRecommender().fit(train)

    assert model.recommend("unknown", k=4) == fallback.recommend("unknown", k=4)
