from __future__ import annotations

import pandas as pd

from .base import Recommender
from .data import assign_segments


class SwitchingHybrid(Recommender):  # S06 slide 21: one RS per user group, chosen by amount of feedback
    name = "switching_hybrid"

    def __init__(self, new_model: Recommender, light_model: Recommender, regular_model: Recommender):
        self.new_model = new_model
        self.light_model = light_model
        self.regular_model = regular_model

    def fit(self, train, articles=None, customers=None):
        self.models_ = {"new": self.new_model, "light": self.light_model, "regular": self.regular_model}
        fitted = set()
        for model in self.models_.values():
            if id(model) not in fitted:
                model.fit(train, articles, customers)
                fitted.add(id(model))
        self.history_ = train[["customer_id"]]
        return self

    def segments(self, customer_ids) -> pd.Series:
        ids = pd.Series([str(c) for c in customer_ids])
        return pd.Series(assign_segments(self.history_, ids).astype(str).to_numpy(), index=ids)

    def recommend_all(self, customer_ids, k=12):
        segs = self.segments(customer_ids)
        out = {}
        for label, model in self.models_.items():
            group = segs.index[segs == label].tolist()
            if group:
                out.update(model.recommend_all(group, k=k))
        return {c: out[c] for c in segs.index}

    def recommend(self, customer_id, k=12):
        return self.recommend_all([customer_id], k)[str(customer_id)]


class WeightedHybrid(Recommender):  # S06 slides 20 + 22: rank points per RS times a weight, equal weights = mixed
    name = "weighted_hybrid"

    def __init__(self, models: list[Recommender], weights: list[float] | None = None, n: int | None = None):
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.n = n

    def fit(self, train, articles=None, customers=None):
        fitted = set()
        for model in self.models:
            if id(model) not in fitted:
                model.fit(train, articles, customers)
                fitted.add(id(model))
        return self

    @staticmethod
    def rank_points(ranked_lists, weights):
        points = {}
        for weight, items in zip(weights, ranked_lists):
            for pos, item in enumerate(items):
                points[item] = points.get(item, 0) + weight * (len(items) - pos)
        return points

    def recommend_all(self, customer_ids, k=12):
        ids = [str(c) for c in customer_ids]
        lists = [model.recommend_all(ids, k=self.n or k) for model in self.models]
        out = {}
        for c in ids:
            points = self.rank_points([rec[c] for rec in lists], self.weights)
            out[c] = sorted(points, key=points.get, reverse=True)[:k]
        return out

    def recommend(self, customer_id, k=12):
        return self.recommend_all([customer_id], k)[str(customer_id)]
