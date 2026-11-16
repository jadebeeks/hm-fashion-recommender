"""Collaborative filtering: item-to-item similarity and ALS matrix factorisation.

Owner: Member 3 (Collaborative Filtering), branch ``m3-collaborative``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, vstack

from implicit.als import AlternatingLeastSquares

from .base import Recommender
from .baselines import PopularityRecommender


class ItemItemCF(Recommender):
    """"Customers who bought X also bought Y", from items bought by the same customers.

    Purchases are binary implicit feedback: repeat transactions by one customer
    do not increase that customer's influence. Similarities are computed one
    item row at a time and truncated to ``n_neighbors``, avoiding a dense
    item-by-item matrix.
    """
    name = "item_item_cf"

    def __init__(self, n_neighbors: int = 100):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        self.n_neighbors = n_neighbors

    def fit(self, train, articles=None, customers=None):
        required = {"customer_id", "article_id"}
        if not required.issubset(train.columns):
            raise ValueError("Training data must contain customer_id and article_id.")
        if train.empty:
            raise ValueError("Training data must contain at least one interaction.")
        if train[["customer_id", "article_id"]].isna().any().any():
            raise ValueError("Training customer and article IDs cannot be missing.")

        customer_ids = train["customer_id"].astype("string")
        customer_codes, unique_customers = pd.factorize(customer_ids, sort=True)
        self.customer_index_ = {
            str(customer_id): index
            for index, customer_id in enumerate(unique_customers)
        }

        article_values = pd.to_numeric(train["article_id"], errors="raise").to_numpy(
            dtype=np.int64
        )
        self.article_ids_, article_codes = np.unique(
            article_values, return_inverse=True
        )

        # Sum duplicate COO coordinates, then replace counts with binary feedback.
        interactions = coo_matrix(
            (
                np.ones(len(train), dtype=np.int32),
                (customer_codes, article_codes),
            ),
            shape=(len(unique_customers), len(self.article_ids_)),
        ).tocsr()
        interactions.data.fill(1)
        self.interaction_matrix_ = interactions.astype(np.float32)
        self.interaction_matrix_.sort_indices()

        item_customers = self.interaction_matrix_.T.tocsr()
        item_support = np.asarray(item_customers.getnnz(axis=1))
        n_items = len(self.article_ids_)

        similarity_rows = []

        for item_index in range(n_items):
            cooccurrences = item_customers.getrow(item_index) @ self.interaction_matrix_
            neighbor_indices = cooccurrences.indices
            not_self = neighbor_indices != item_index
            neighbor_indices = neighbor_indices[not_self]
            cooccurrence_counts = cooccurrences.data[not_self]

            if len(neighbor_indices):
                similarities = cooccurrence_counts / np.sqrt(
                    item_support[item_index] * item_support[neighbor_indices]
                )
                order = np.lexsort(
                    (self.article_ids_[neighbor_indices], -similarities)
                )[: self.n_neighbors]
                neighbor_indices = neighbor_indices[order].astype(np.int32, copy=False)
                similarities = similarities[order].astype(np.float32, copy=False)
            else:
                similarities = np.empty(0, dtype=np.float32)

            similarity_rows.append(
                coo_matrix(
                    (
                        similarities,
                        (np.zeros(len(neighbor_indices), dtype=np.int32), neighbor_indices),
                    ),
                    shape=(1, n_items),
                ).tocsr()
            )

        self.similarity_matrix_ = vstack(
            similarity_rows, format="csr", dtype=np.float32
        )

        self.fallback_ = PopularityRecommender().fit(train)
        return self

    def recommend(self, customer_id, k=12):
        if k < 0:
            raise ValueError("k must be non-negative.")
        if k == 0:
            return []

        customer_index = self.customer_index_.get(str(customer_id))
        purchased: set[int] = set()
        ranked: list[int] = []

        if customer_index is not None:
            history = self.interaction_matrix_.getrow(customer_index)
            purchased = set(self.article_ids_[history.indices].tolist())
            scores = history @ self.similarity_matrix_
            candidate_indices = scores.indices
            candidate_scores = scores.data
            eligible = np.array(
                [
                    self.article_ids_[index] not in purchased
                    for index in candidate_indices
                ],
                dtype=bool,
            )
            candidate_indices = candidate_indices[eligible]
            candidate_scores = candidate_scores[eligible]
            order = np.lexsort(
                (self.article_ids_[candidate_indices], -candidate_scores)
            )
            ranked = self.article_ids_[candidate_indices[order]].astype(int).tolist()

        recommendations = ranked[:k]
        if len(recommendations) < k:
            seen = purchased | set(recommendations)
            for article_id in self.fallback_.recommend(
                customer_id, k=len(self.fallback_.ranking_)
            ):
                article_id = int(article_id)
                if article_id not in seen:
                    recommendations.append(article_id)
                    seen.add(article_id)
                    if len(recommendations) == k:
                        break

        return recommendations


class ALSRecommender(Recommender):
    """Matrix factorisation for purchase data (``implicit`` library).

    Each distinct purchase is a positive implicit interaction. ``alpha=40``
    gives observed interactions confidence 40 while unobserved pairs retain
    the library's baseline confidence of 1.
    """
    name = "als"

    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations

    def fit(self, train, articles=None, customers=None):
        required = {"customer_id", "article_id"}
        if not required.issubset(train.columns):
            raise ValueError("Training data must contain customer_id and article_id.")
        if train.empty:
            raise ValueError("Training data must contain at least one interaction.")
        if train[["customer_id", "article_id"]].isna().any().any():
            raise ValueError("Training customer and article IDs cannot be missing.")

        customer_ids = train["customer_id"].astype("string")
        customer_codes, unique_customers = pd.factorize(customer_ids, sort=True)
        self.customer_index_ = {
            str(customer_id): index
            for index, customer_id in enumerate(unique_customers)
        }

        article_values = pd.to_numeric(train["article_id"], errors="raise").to_numpy(
            dtype=np.int64
        )
        self.article_ids_, article_codes = np.unique(
            article_values, return_inverse=True
        )

        interactions = coo_matrix(
            (
                np.ones(len(train), dtype=np.float32),
                (customer_codes, article_codes),
            ),
            shape=(len(unique_customers), len(self.article_ids_)),
        ).tocsr()
        interactions.data.fill(1.0)
        self.interaction_matrix_ = interactions
        self.interaction_matrix_.sort_indices()

        self.model_ = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            alpha=40.0,
            random_state=42,
        )
        self.model_.fit(self.interaction_matrix_, show_progress=False)
        self.fallback_ = PopularityRecommender().fit(train)
        return self

    def recommend(self, customer_id, k=12):
        if k < 0:
            raise ValueError("k must be non-negative.")
        if k == 0:
            return []

        customer_index = self.customer_index_.get(str(customer_id))
        if customer_index is None:
            return self.fallback_.recommend(customer_id, k)

        history = self.interaction_matrix_.getrow(customer_index)
        purchased = set(self.article_ids_[history.indices].tolist())
        available_count = len(self.article_ids_) - len(purchased)
        if available_count:
            item_indices, _ = self.model_.recommend(
                customer_index,
                history,
                N=min(k, available_count),
                filter_already_liked_items=True,
            )
        else:
            item_indices = []

        recommendations: list[int] = []
        seen = purchased.copy()
        for item_index in item_indices:
            if item_index < 0 or item_index >= len(self.article_ids_):
                continue
            article_id = int(self.article_ids_[item_index])
            if article_id not in seen:
                recommendations.append(article_id)
                seen.add(article_id)
                if len(recommendations) == k:
                    return recommendations

        for article_id in self.fallback_.recommend(
            customer_id, k=len(self.fallback_.ranking_)
        ):
            article_id = int(article_id)
            if article_id not in seen:
                recommendations.append(article_id)
                seen.add(article_id)
                if len(recommendations) == k:
                    break

        return recommendations
