"""Checks for Member 4's content-based recommender on a tiny made-up catalogue. Run with: pytest"""
import numpy as np
import pandas as pd
import pytest

from hm_recsys.content import ContentBasedRecommender, build_item_text, vectorize


@pytest.fixture
def data():
    articles = pd.DataFrame({
        "article_id": [101, 102, 103, 201, 202, 203, 301, 302, 401, 501],
        "product_code": [1, 1, 2, 3, 3, 4, 5, 6, 7, 8],
        "prod_name": ["Basic tee", "Basic tee", "Slim tee", "Denim skinny", "Denim skinny",
                      "Wide denim", "Summer dress", "Maxi dress", "Wool cardigan", "Rain jacket"],
        "product_type_name": ["T-shirt", "T-shirt", "T-shirt", "Trousers", "Trousers",
                              "Trousers", "Dress", "Dress", "Cardigan", "Jacket"],
        "colour_group_name": ["Black", "White", "Black", "Blue", "Black", "Blue",
                              "Red", "Unknown", "Grey", "Yellow"],
        "detail_desc": ["Cotton jersey t-shirt with short sleeves",
                        "Cotton jersey t-shirt with short sleeves",
                        "Fitted cotton jersey t-shirt",
                        "Skinny jeans in washed stretch denim",
                        "Skinny jeans in washed stretch denim",
                        "Wide jeans in sturdy denim",
                        "Short dress in airy woven fabric",
                        "Long dress in airy woven fabric",
                        "Cardigan in soft knitted wool",
                        None],
    })
    rows = [
        ("2020-09-01", "tee_fan", 101),       # bought a tee long ago...
        ("2020-09-14", "tee_fan", 102),       # ...and again recently
        ("2020-09-01", "mixed", 301),         # dress long ago
        ("2020-09-15", "mixed", 201),         # jeans yesterday -> recency should favour denim
        ("2020-09-15", "x", 103), ("2020-09-15", "y", 202), ("2020-09-15", "z", 203),
        ("2020-09-15", "x", 302), ("2020-09-14", "y", 401), ("2020-09-14", "z", 501),
    ]
    train = pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"])
    train["t_dat"] = pd.to_datetime(train["t_dat"])
    return train, articles


def test_item_text_joins_parts_and_drops_placeholders(data):
    _, articles = data
    text = build_item_text(articles)
    assert list(text.index) == articles["article_id"].tolist()
    assert "Basic tee" in text[101] and "Black" in text[101] and "short sleeves" in text[101]
    assert "Unknown" not in text[302]
    assert text[501] == "Rain jacket. Jacket Yellow"          # missing description is fine
    only_desc = build_item_text(articles, fields=["description"])
    assert only_desc[101] == "Cotton jersey t-shirt with short sleeves"
    with pytest.raises(ValueError):
        build_item_text(articles, fields=["price"])


@pytest.mark.parametrize("method", ["bow", "tfidf"])
def test_vectors_are_unit_length(data, method):
    _, articles = data
    X, _ = vectorize(build_item_text(articles), method, min_df=1)
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    assert np.allclose(norms, 1.0, atol=1e-5)


MODELS = [
    lambda: ContentBasedRecommender(min_df=1, candidate_weeks=None),
    lambda: ContentBasedRecommender("bow", min_df=1, candidate_weeks=1),
    lambda: ContentBasedRecommender(min_df=1, popularity_weight=0.3, one_per_product=True),
    lambda: ContentBasedRecommender(min_df=1, candidate_items=[501],
                                    only_candidate_items=True),
]


@pytest.mark.parametrize("make", MODELS)
@pytest.mark.parametrize("cid", ["tee_fan", "mixed", "never-seen-id"])
def test_always_k_distinct_ints_for_any_customer(data, make, cid):
    train, articles = data
    model = make().fit(train, articles)
    recs = model.recommend(cid, k=5)
    assert len(recs) == 5 and len(set(recs)) == 5
    assert all(isinstance(i, int) for i in recs)
    assert model.recommend_all([cid], k=5)[cid] == recs


def test_recommends_similar_unbought_items(data):
    train, articles = data
    model = ContentBasedRecommender(min_df=1, candidate_weeks=None).fit(train, articles)
    recs = model.recommend("tee_fan", k=3)
    assert 103 == recs[0]                       # the other tee comes first
    assert not {101, 102} & set(recs)           # already bought


def test_recency_weighting_changes_the_order(data):
    train, articles = data
    recent = ContentBasedRecommender(recency_half_life_days=1, min_df=1,
                                     candidate_weeks=None).fit(train, articles)
    flat = ContentBasedRecommender(recency_half_life_days=None, min_df=1,
                                   candidate_weeks=None).fit(train, articles)
    assert recent.recommend("mixed", 1)[0] in {202, 203}       # jeans bought yesterday win
    w = recent.history("mixed").set_index("article_id")["weight"]
    assert w[201] > w[301]
    assert set(flat.history("mixed")["weight"]) == {1.0}


def test_candidate_pool_limits_to_recent_sellers(data):
    train, articles = data
    model = ContentBasedRecommender(min_df=1, candidate_weeks=1).fit(train, articles)
    recent = set(train.loc[train["t_dat"] > "2020-09-08", "article_id"])
    assert set(model.pool_ids_) == recent


def test_one_per_product_drops_colour_variants(data):
    train, articles = data
    model = ContentBasedRecommender(min_df=1, candidate_weeks=None,
                                    exclude_purchased=False, one_per_product=True).fit(train, articles)
    codes = articles.set_index("article_id")["product_code"]
    recs = model.recommend("tee_fan", k=4)
    assert codes[recs].is_unique


def test_launch_path_only_returns_launches_first(data):
    train, articles = data
    launches = [301, 302, 401]
    model = ContentBasedRecommender(min_df=1, candidate_items=launches,
                                    only_candidate_items=True,
                                    exclude_purchased=False).fit(train, articles)
    assert set(model.recommend("mixed", k=3)) == set(launches)
    assert set(model.recommend("never-seen-id", k=3)) == set(launches)
    with pytest.raises(ValueError):
        ContentBasedRecommender(only_candidate_items=True).fit(train, articles)


def test_audience_puts_the_closest_profile_first(data):
    train, articles = data
    model = ContentBasedRecommender(min_df=1).fit(train, articles)
    aud = model.audience([101, 203], n=2)
    assert aud[101][0] == "tee_fan"
    assert aud[203][0] in {"mixed", "y", "z"}
    assert len(aud[101]) == 2


def test_explain_and_similar_items(data):
    train, articles = data
    model = ContentBasedRecommender(min_df=1, candidate_weeks=None).fit(train, articles)
    why = model.explain("mixed", 202, top=1)
    assert why["article_id"].iloc[0] == 201
    sim = model.similar_items(201, k=2)
    assert 202 in sim.index and 201 not in sim.index
    assert model.explain("never-seen-id", 202).empty


def test_precomputed_vectors_work(data):
    train, articles = data
    rng = np.random.default_rng(0)
    vecs = pd.DataFrame(rng.normal(size=(len(articles), 8)), index=articles["article_id"])
    model = ContentBasedRecommender(item_vectors=vecs, candidate_weeks=None).fit(train, articles)
    recs = model.recommend("tee_fan", k=5)
    assert len(set(recs)) == 5
    assert model.name == "content_vectors"


def test_works_with_shared_evaluate(data):
    from hm_recsys.evaluation import evaluate
    train, articles = data
    test = pd.DataFrame({"customer_id": ["tee_fan", "mixed", "brand_new"],
                         "article_id": [103, 202, 101]})
    model = ContentBasedRecommender(min_df=1).fit(train, articles)
    res = evaluate(model, test, k=3, train=train)
    assert {"all", "new", "light"} <= set(res.index)
    assert res.loc["all", "HitRate@3"] > 0


def test_audience_respects_universe(data):
    train, articles = data
    model = ContentBasedRecommender(min_df=1).fit(train, articles)
    aud = model.audience([101], n=5, universe=["x", "y"])
    assert set(aud[101]) == {"x", "y"}


def test_attribute_popularity_ranks_launches_by_selling_type(data):
    from hm_recsys.content import AttributePopularityRecommender
    train, articles = data
    # in the last week: T-shirt (103, x), Trousers (202, 203: y, z), Dress (302, x)
    model = AttributePopularityRecommender([101, 201, 301], weeks=1,
                                           spread=False).fit(train, articles)
    recs = model.recommend("anyone", k=3)
    assert recs[0] == 201                       # trousers sold to 2 customers
    assert set(recs) == {101, 201, 301}
    # spread: one launch per type before a second one of the same type
    spread = AttributePopularityRecommender([201, 202, 203, 101], weeks=1).fit(train, articles)
    first_two = articles.set_index("article_id").loc[spread.recommend("anyone", k=2),
                                                     "product_type_name"]
    assert set(first_two) == {"Trousers", "T-shirt"}


def test_lemmatized_model_runs_if_spacy_is_installed(data):
    pytest.importorskip("spacy")
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    train, articles = data
    model = ContentBasedRecommender(lemmatize=True, min_df=1).fit(train, articles)
    assert "pocket" not in model.texts_[101]     # no pockets in this text, just a smoke test
    assert len(set(model.recommend("tee_fan", k=4))) == 4
