"""Non-personalised and demographic recommenders, plus the new-customer path.

Owner: Member 2 (Baselines & New Customers), branch ``m2-baselines``.

All models here depend only on ``base.Recommender`` and plain pandas, so the
other members can use them as fallbacks before the rest of the pipeline exists.

Expected columns
----------------
train      : ``t_dat`` (date), ``customer_id`` (str), ``article_id`` (int)
customers  : ``customer_id``, ``age`` (may be missing)
articles   : ``article_id`` (used for the random pool and the bias analysis)

Quick use
---------
>>> trending = PopularityRecommender(weeks=1).fit(train)
>>> new_path = NewCustomerRecommender(weeks=1).fit(train, customers=customers)
>>> new_path.recommend("some-unknown-id")   # always returns 12 items
"""
from __future__ import annotations

import zlib
from typing import Iterable

import numpy as np
import pandas as pd

from .base import Recommender

# Age bands used for demographic filtering. Ten-year bands from 25 up; the
# youngest H&M customers (16-24) are a large, distinct group, so they get their own band.
AGE_BINS = [0, 25, 35, 45, 55, 65, 200]
AGE_LABELS = ["16-24", "25-34", "35-44", "45-54", "55-64", "65+"]
UNKNOWN_AGE = "unknown"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _last_weeks(train: pd.DataFrame, weeks: int | None) -> pd.DataFrame:
    """Rows from the final ``weeks`` weeks of ``train`` (all rows if ``weeks`` is None)."""
    if weeks is None:
        return train
    dates = pd.to_datetime(train["t_dat"])
    cutoff = dates.max() - pd.Timedelta(weeks=weeks)
    return train[dates > cutoff]


def _rank_items(df: pd.DataFrame, count_by: str = "customers") -> list[int]:
    """Article ids from most to least popular.

    ``count_by="customers"`` counts distinct buyers, so one shopper buying ten
    of the same item does not make it "popular". ``"rows"`` counts transactions.
    """
    if df.empty:
        return []
    if count_by == "customers":
        counts = df.groupby("article_id", observed=True)["customer_id"].nunique()
    elif count_by == "rows":
        counts = df["article_id"].value_counts()
    else:
        raise ValueError("count_by must be 'customers' or 'rows'")
    # stable tie-break on article_id so results are reproducible
    counts = counts.rename("n").reset_index().sort_values(
        ["n", "article_id"], ascending=[False, True])
    return counts["article_id"].astype(int).tolist()


def _top_up(items: list[int], fallback: Iterable[int], k: int) -> list[int]:
    """Fill ``items`` up to ``k`` with unseen items from ``fallback``, keeping order."""
    out = list(dict.fromkeys(items))[:k]
    if len(out) < k:
        seen = set(out)
        for item in fallback:
            if item not in seen:
                out.append(item)
                seen.add(item)
                if len(out) == k:
                    break
    return out


def age_band(age: pd.Series) -> pd.Series:
    """Map ages to the bands in ``AGE_LABELS``; missing ages become ``'unknown'``."""
    bands = pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return bands.astype(object).where(bands.notna(), UNKNOWN_AGE)


def customer_age_bands(customers: pd.DataFrame) -> pd.Series:
    """Age band per customer, indexed by ``customer_id``. Handy for ``evaluate(segments=...)``."""
    return pd.Series(age_band(customers["age"]).to_numpy(),
                     index=customers["customer_id"], name="age_band")


# ---------------------------------------------------------------------------
# 1. Random: what the company does today
# ---------------------------------------------------------------------------
class RandomRecommender(Recommender):
    """Random products: the company's current display strategy.

    The pool is the whole catalogue when ``articles`` is given (what a random
    display really shows), otherwise the items seen in ``train``.
    The draw is seeded per customer, so results are reproducible.
    """
    name = "random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, train, articles=None, customers=None):
        if articles is not None:
            pool = articles["article_id"]
        else:
            pool = train["article_id"]
        self.pool_ = np.unique(pool.to_numpy()).astype(np.int64)
        return self

    def recommend(self, customer_id, k=12):
        rng = np.random.default_rng([self.seed, zlib.crc32(str(customer_id).encode())])
        size = min(k, len(self.pool_))
        return rng.choice(self.pool_, size=size, replace=False).tolist()


# ---------------------------------------------------------------------------
# 2 + 3. Global popularity and "trending now"
# ---------------------------------------------------------------------------
class PopularityRecommender(Recommender):
    """Most-bought items, over all of train or only the last ``weeks`` weeks ("trending now").

    Everyone gets the same list. ``weeks=None`` gives all-time best sellers;
    ``weeks=1`` or ``2`` gives "trending now", which reacts to seasons and new
    collections and is the baseline every personalised model has to beat.
    """

    def __init__(self, weeks: int | None = None, count_by: str = "customers"):
        self.weeks = weeks
        self.count_by = count_by
        self.name = "popularity" if weeks is None else f"trending_{weeks}w"

    def fit(self, train, articles=None, customers=None):
        self.recent_ = _rank_items(_last_weeks(train, self.weeks), self.count_by)
        # always-available fallback in case the window is very short
        self.ranking_ = _top_up(self.recent_, _rank_items(train, self.count_by),
                                k=max(len(self.recent_), 1000))
        return self

    def recommend(self, customer_id, k=12):
        return self.ranking_[:k]

    def recommend_all(self, customer_ids, k=12):
        top = self.ranking_[:k]
        return {c: list(top) for c in customer_ids}


# ---------------------------------------------------------------------------
# 4. Demographic filtering, which is also the new-customer path
# ---------------------------------------------------------------------------
class AgeBandPopularityRecommender(Recommender):
    """Demographic filtering: trending items within the customer's age band.

    - age known      -> best sellers of the last ``weeks`` weeks among people of that age band;
    - age unknown,
      or customer not
      in ``customers`` -> global "trending now".
    Short band lists are topped up with global trends, so every customer gets ``k`` items.
    This is the new-customer path used by the hybrid.
    """
    name = "popularity_by_age"

    def __init__(self, weeks: int | None = 1, count_by: str = "customers"):
        self.weeks = weeks
        self.count_by = count_by

    def fit(self, train, articles=None, customers=None):
        if customers is None:
            raise ValueError("AgeBandPopularityRecommender needs the customers table (for age)")
        self.global_ = PopularityRecommender(self.weeks, self.count_by).fit(train)
        self.band_of_ = customer_age_bands(customers).to_dict()

        recent = _last_weeks(train, self.weeks)
        recent = recent.assign(age_band=recent["customer_id"].map(self.band_of_).fillna(UNKNOWN_AGE))
        global_list = self.global_.ranking_
        self.band_ranking_ = {
            band: _top_up(_rank_items(grp, self.count_by), global_list, k=100)
            for band, grp in recent.groupby("age_band")
            if band != UNKNOWN_AGE
        }
        return self

    def band(self, customer_id) -> str:
        """The age band used for this customer (``'unknown'`` if no age)."""
        return self.band_of_.get(customer_id, UNKNOWN_AGE)

    def recommend(self, customer_id, k=12):
        ranking = self.band_ranking_.get(self.band(customer_id))
        if ranking is None:
            return self.global_.recommend(customer_id, k)
        return _top_up(ranking[:k], self.global_.ranking_, k)

    def recommend_all(self, customer_ids, k=12):
        cache = {b: self.recommend_band(b, k) for b in [*self.band_ranking_, UNKNOWN_AGE]}
        return {c: list(cache.get(self.band(c), cache[UNKNOWN_AGE])) for c in customer_ids}

    def recommend_band(self, band: str, k: int = 12) -> list[int]:
        """The list shown to everyone in ``band`` (useful for the demo slides)."""
        ranking = self.band_ranking_.get(band)
        if ranking is None:
            return self.global_.ranking_[:k]
        return _top_up(ranking[:k], self.global_.ranking_, k)


class NewCustomerRecommender(AgeBandPopularityRecommender):
    """The new-customer path for the hybrid (Member 5).

    Same logic as ``AgeBandPopularityRecommender``: age-band trends when age is
    known, global trends when it isn't. It has its own name so the hybrid and
    the slides can refer to it directly.
    """
    name = "new_customer_path"


# ---------------------------------------------------------------------------
# 5. Buy again
# ---------------------------------------------------------------------------
class BuyAgainRecommender(Recommender):
    """The customer's own most frequently bought items, topped up with trends.

    Items are ranked by how many times the customer bought them, then by how
    recently. If ``weeks`` is set, only purchases in that window count (fashion
    goes out of stock quickly). Remaining slots come from ``fallback``, by
    default the new-customer path, so customers we know nothing about still get a full list.
    """
    name = "buy_again"

    def __init__(self, weeks: int | None = None, fallback: Recommender | None = None,
                 fallback_weeks: int = 1):
        self.weeks = weeks
        self.fallback = fallback
        self.fallback_weeks = fallback_weeks

    def fit(self, train, articles=None, customers=None):
        if self.fallback is None:
            self.fallback = (NewCustomerRecommender(self.fallback_weeks) if customers is not None
                             else PopularityRecommender(self.fallback_weeks))
        self.fallback.fit(train, articles, customers)

        hist = _last_weeks(train, self.weeks)
        agg = (hist.groupby(["customer_id", "article_id"], observed=True)
                   .agg(n=("t_dat", "size"), last=("t_dat", "max"))
                   .reset_index()
                   .sort_values(["customer_id", "n", "last", "article_id"],
                                ascending=[True, False, False, True]))
        self.history_ = (agg.groupby("customer_id", observed=True)["article_id"]
                            .agg(lambda s: s.astype(int).tolist())
                            .to_dict())
        return self

    def recommend(self, customer_id, k=12):
        own = self.history_.get(customer_id, [])[:k]
        if len(own) == k:
            return own
        return _top_up(own, self.fallback.recommend(customer_id, 2 * k), k)


# ---------------------------------------------------------------------------
# Popularity-bias analysis
# ---------------------------------------------------------------------------
def exposure_table(predicted: dict[str, list[int]], train: pd.DataFrame,
                   articles: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per article sold in train: units sold, times recommended, and attributes.

    Used to answer "which items never get shown?". ``articles`` adds product
    group / type columns when given.
    """
    sold = train["article_id"].value_counts().rename("sold")
    shown = pd.Series(
        [i for recs in predicted.values() for i in recs], dtype="int64"
    ).value_counts().rename("shown")
    tab = pd.concat([sold, shown], axis=1).fillna(0).astype(int)
    tab.index.name = "article_id"
    tab = tab.reset_index()
    if articles is not None:
        cols = [c for c in ["article_id", "prod_name", "product_type_name", "product_group_name",
                            "index_group_name", "colour_group_name"] if c in articles.columns]
        tab = tab.merge(articles[cols], on="article_id", how="left")
    return tab.sort_values("sold", ascending=False, ignore_index=True)


def popularity_bias_summary(predicted: dict[str, list[int]], train: pd.DataFrame,
                            top_share: float = 0.01) -> dict[str, float]:
    """Plain-language numbers on how concentrated a model's recommendations are.

    - ``distinct_items_shown``: how many different products anyone sees;
    - ``share_slots_top_items``: share of all recommendation slots taken by the top
      ``top_share`` best-selling items;
    - ``share_sold_items_never_shown``: share of items that sold in train but are never recommended;
    - ``share_sales_never_shown``: share of train sales coming from those never-shown items.
    The Gini index itself comes from ``evaluate()`` (Member 1).
    """
    tab = exposure_table(predicted, train)
    sold_items = tab[tab["sold"] > 0]
    n_top = max(1, int(round(top_share * len(sold_items))))
    top_items = set(sold_items.head(n_top)["article_id"])
    total_slots = tab["shown"].sum()
    never = sold_items[sold_items["shown"] == 0]
    return {
        "distinct_items_shown": int((tab["shown"] > 0).sum()),
        "share_slots_top_items": float(tab.loc[tab["article_id"].isin(top_items), "shown"].sum()
                                       / max(total_slots, 1)),
        "share_sold_items_never_shown": float(len(never) / max(len(sold_items), 1)),
        "share_sales_never_shown": float(never["sold"].sum() / max(sold_items["sold"].sum(), 1)),
    }


def lorenz_curve(predicted: dict[str, list[int]], n_items: int) -> tuple[np.ndarray, np.ndarray]:
    """Points for a Lorenz curve of recommendation exposure over ``n_items`` catalogue items.

    x = cumulative share of items (least to most shown), y = cumulative share of slots.
    A diagonal line means every item is shown equally; a curve hugging the
    bottom-right corner means a handful of items take all the exposure.
    """
    counts = pd.Series([i for r in predicted.values() for i in r]).value_counts().to_numpy()
    counts = np.concatenate([np.zeros(max(n_items - len(counts), 0)), np.sort(counts)])
    y = np.concatenate([[0], np.cumsum(counts) / max(counts.sum(), 1)])
    x = np.linspace(0, 1, len(y))
    return x, y
