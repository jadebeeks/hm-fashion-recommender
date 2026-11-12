"""Shared interface for every recommender in the project.

Owner: whole team. Change this file only after everyone agrees,
because all models and the evaluation depend on it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd


class Recommender(ABC):
    """Base class for all recommenders.

    Every subclass must:
    - learn from the training transactions in ``fit``;
    - return exactly ``k`` article_ids for ANY customer_id in ``recommend``,
      including customers it has never seen (use a fallback such as popularity).
    """

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        train: pd.DataFrame,
        articles: pd.DataFrame | None = None,
        customers: pd.DataFrame | None = None,
    ) -> "Recommender":
        """Learn from training data. Must return ``self``."""

    @abstractmethod
    def recommend(self, customer_id: str, k: int = 12) -> list[int]:
        """Return a ranked list of ``k`` article_ids for one customer."""

    def recommend_all(self, customer_ids: Iterable[str], k: int = 12) -> dict[str, list[int]]:
        """Recommend for many customers. Override if your model can batch faster."""
        return {c: self.recommend(c, k) for c in customer_ids}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
