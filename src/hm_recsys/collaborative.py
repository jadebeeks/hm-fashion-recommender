"""Collaborative filtering: item-to-item similarity and ALS matrix factorisation.

Owner: Member 3 (Collaborative Filtering), branch ``m3-collaborative``.
"""
from __future__ import annotations

from .base import Recommender


class ItemItemCF(Recommender):
    """"Customers who bought X also bought Y", from items bought by the same customers.

    TODO (Member 3): build a sparse customer x item matrix, compute item-item
    similarity (e.g. cosine), score items from each customer's history.
    Unknown customers -> fallback (e.g. Member 2's popularity).
    """
    name = "item_item_cf"

    def fit(self, train, articles=None, customers=None):
        raise NotImplementedError

    def recommend(self, customer_id, k=12):
        raise NotImplementedError


class ALSRecommender(Recommender):
    """Matrix factorisation for purchase data (``implicit`` library).

    TODO (Member 3): light tuning only (factors, regularization).
    """
    name = "als"

    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, train, articles=None, customers=None):
        raise NotImplementedError

    def recommend(self, customer_id, k=12):
        raise NotImplementedError
