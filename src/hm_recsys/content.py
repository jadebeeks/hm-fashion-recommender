"""Content-based recommender and the new-item path.

Owner: Member 4 (Content-Based & New Products), branch ``m4-content``.
"""
from __future__ import annotations

from .base import Recommender


def build_item_text(articles):
    """Join description + attributes (product type, colour, pattern, department) into one text per item.

    TODO (Member 4)
    """
    raise NotImplementedError


class ContentBasedRecommender(Recommender):
    """TF-IDF (or BoW) item vectors; customer profile = recency-weighted mean of purchased items.

    Also handles new items: they can be recommended as soon as they have a description.
    TODO (Member 4)
    """
    name = "content_tfidf"

    def __init__(self, vectorizer: str = "tfidf", recency_half_life_days: float = 14.0):
        self.vectorizer = vectorizer
        self.recency_half_life_days = recency_half_life_days

    def fit(self, train, articles=None, customers=None):
        raise NotImplementedError

    def recommend(self, customer_id, k=12):
        raise NotImplementedError
