"""The final system: picks the right strategy for every customer.

Owner: Member 5 (Hybrid, Business Value & Presentation Lead), branch ``m5-hybrid-business``.
"""
from __future__ import annotations

from .base import Recommender


class SwitchingHybrid(Recommender):
    """Route each customer by segment:

    - new (0 purchases)   -> new-customer path (Member 2)
    - light (1-4)         -> content-based + trends (Members 4 + 2)
    - regular (5+)        -> collaborative filtering (Member 3)

    TODO (Member 5)
    """
    name = "switching_hybrid"

    def __init__(self, new_model: Recommender, light_model: Recommender, regular_model: Recommender):
        self.new_model = new_model
        self.light_model = light_model
        self.regular_model = regular_model

    def fit(self, train, articles=None, customers=None):
        raise NotImplementedError

    def recommend(self, customer_id, k=12):
        raise NotImplementedError


class WeightedHybrid(Recommender):
    """Blend scores from several models with weights (try for regular customers).

    TODO (Member 5)
    """
    name = "weighted_hybrid"

    def __init__(self, models: list[Recommender], weights: list[float]):
        self.models = models
        self.weights = weights

    def fit(self, train, articles=None, customers=None):
        raise NotImplementedError

    def recommend(self, customer_id, k=12):
        raise NotImplementedError
