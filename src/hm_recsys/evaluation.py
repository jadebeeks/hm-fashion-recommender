"""One scoring function for every model.

Owner: Member 1 (Data & Evaluation Lead), branch ``m1-data-eval``.
Other members must use ``evaluate()`` and not write their own scoring.
"""
from __future__ import annotations

import pandas as pd

from .base import Recommender

K = 12  # H&M recommends 12 items per customer


# ---- accuracy (ranking) ----------------------------------------------------
def average_precision_at_k(actual: set[int], predicted: list[int], k: int = K) -> float:
    """AP@k for one customer. TODO (Member 1)"""
    raise NotImplementedError


def map_at_k(actual: dict[str, set[int]], predicted: dict[str, list[int]], k: int = K) -> float:
    """Mean AP@k over customers with at least one test purchase. TODO (Member 1)"""
    raise NotImplementedError


def recall_at_k(actual: dict[str, set[int]], predicted: dict[str, list[int]], k: int = K) -> float:
    """TODO (Member 1)"""
    raise NotImplementedError


def ndcg_at_k(actual: dict[str, set[int]], predicted: dict[str, list[int]], k: int = K) -> float:
    """TODO (Member 1)"""
    raise NotImplementedError


def hit_rate_at_k(actual: dict[str, set[int]], predicted: dict[str, list[int]], k: int = K) -> float:
    """Share of customers with at least one correct item. TODO (Member 1)"""
    raise NotImplementedError


# ---- beyond accuracy -------------------------------------------------------
def catalog_coverage(predicted: dict[str, list[int]], n_items: int) -> float:
    """Share of the catalogue that gets recommended at least once. TODO (Member 1)"""
    raise NotImplementedError


def gini_index(predicted: dict[str, list[int]]) -> float:
    """How concentrated recommendations are on a few items (0 = even, 1 = one item). TODO (Member 1)"""
    raise NotImplementedError


def intra_list_diversity(predicted: dict[str, list[int]], articles: pd.DataFrame,
                         attribute: str = "product_group_name") -> float:
    """Average share of distinct product groups in each list. TODO (Member 1)"""
    raise NotImplementedError


def novelty(predicted: dict[str, list[int]], item_popularity: pd.Series) -> float:
    """Mean self-information (-log2 popularity) of recommended items. TODO (Member 1)"""
    raise NotImplementedError


# ---- main entry point ------------------------------------------------------
def evaluate(model: Recommender, test: pd.DataFrame, segments: pd.Series | None = None,
             k: int = K) -> pd.DataFrame:
    """Score a fitted model on the test week.

    Returns one row per segment ('all', 'new', 'light', 'regular') and
    one column per metric. TODO (Member 1)
    """
    raise NotImplementedError
