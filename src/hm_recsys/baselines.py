"""Non-personalised and demographic recommenders, plus the new-customer path.

Owner: Member 2 (Baselines & New Customers), branch ``m2-baselines``.
"""
from __future__ import annotations

from .base import Recommender


class RandomRecommender(Recommender):
    """What the company does today. TODO (Member 2)"""
    name = "random"

    def fit(self, train, articles=None, customers=None):
        raise NotImplementedError

    def recommend(self, customer_id, k=12):
        raise NotImplementedError


class PopularityRecommender(Recommender):
    """Most-bought items, over all of train or only the last ``weeks`` weeks ("trending now").

    TODO (Member 2)
    """
    name = "popularity"

    def __init__(self, weeks: int | None = None):
        self.weeks = weeks

    def fit(self, train, articles=None, customers=None):
        raise NotImplementedError

    def recommend(self, customer_id, k=12):
        raise NotImplementedError


class AgeBandPopularityRecommender(Recommender):
    """Demographic filtering: trending items within the customer's age band.

    Falls back to global trends if age is unknown. This is the new-customer path.
    TODO (Member 2)
    """
    name = "popularity_by_age"

    def fit(self, train, articles=None, customers=None):
        raise NotImplementedError

    def recommend(self, customer_id, k=12):
        raise NotImplementedError


class BuyAgainRecommender(Recommender):
    """The customer's own most frequently bought items, topped up with trends.

    TODO (Member 2)
    """
    name = "buy_again"

    def fit(self, train, articles=None, customers=None):
        raise NotImplementedError

    def recommend(self, customer_id, k=12):
        raise NotImplementedError
