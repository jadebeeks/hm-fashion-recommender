"""One scoring function for every model.

Owner: Member 1 (Data & Evaluation Lead), branch ``m1-data-eval``.
Other members must use ``evaluate()`` and not write their own scoring.
"""
from __future__ import annotations

import pandas as pd

from .base import Recommender

import numpy as np

from collections import Counter
K = 12

def average_precision_at_k(actual: set[int], predicted: list[int], k: int = K) -> float:

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not actual:
        return 0.0

    score = 0.0
    hits = 0
    seen = set()

    for rank, item in enumerate(predicted[:k], start=1):

        if item in actual and item not in seen:
            hits += 1
            score += hits / rank

        seen.add(item)

    return score / min(len(actual), k)


def map_at_k(actual: dict[str, set[int]], predicted: dict[str, list[int]], k: int = K) -> float:

    if k < 1:
        raise ValueError("k must be at least 1.")

    customers = [customer_id for customer_id, items in actual.items() if items]

    if not customers:
        return 0.0

    scores = [
        average_precision_at_k(
            actual[customer_id],
            predicted.get(customer_id, []),
            k
        )
        for customer_id in customers
    ]

    return sum(scores) / len(scores)


def recall_at_k(actual: dict[str, set[int]], predicted: dict[str, list[int]], k: int = K) -> float:

    if k < 1:
        raise ValueError("k must be at least 1.")

    customers = [customer_id for customer_id, items in actual.items() if items]

    if not customers:
        return 0.0

    scores = []

    for customer_id in customers:
        relevant_items = actual[customer_id]
        recommended_items = set(predicted.get(customer_id, [])[:k])

        hits = len(relevant_items & recommended_items)
        scores.append(hits / len(relevant_items))

    return sum(scores) / len(scores)


def ndcg_at_k(actual: dict[str, set[int]], predicted: dict[str, list[int]], k: int = K) -> float:

    if k < 1:
        raise ValueError("k must be at least 1.")

    customers = [customer_id for customer_id, items in actual.items() if items]

    if not customers:
        return 0.0

    scores = []

    for customer_id in customers:
        relevant_items = actual[customer_id]
        recommendations = predicted.get(customer_id, [])[:k]

        dcg = 0.0
        seen = set()

        for rank, item in enumerate(recommendations, start=1):
            if item in relevant_items and item not in seen:
                dcg += 1 / np.log2(rank + 1)

            seen.add(item)

        ideal_hits = min(len(relevant_items), k)

        idcg = sum(
            1 / np.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )

        scores.append(dcg / idcg)

    return sum(scores) / len(scores)


def hit_rate_at_k(actual: dict[str, set[int]], predicted: dict[str, list[int]], k: int = K) -> float:

    if k < 1:
        raise ValueError("k must be at least 1.")

    customers = [customer_id for customer_id, items in actual.items() if items]

    if not customers:
        return 0.0

    hits = 0

    for customer_id in customers:
        relevant_items = actual[customer_id]
        recommended_items = set(predicted.get(customer_id, [])[:k])

        if relevant_items & recommended_items:
            hits += 1

    return hits / len(customers)


def catalog_coverage(predicted: dict[str, list[int]], n_items: int) -> float:

    if n_items <= 0:
        raise ValueError("n_items must be greater than zero.")

    recommended_items = {
        item
        for recommendations in predicted.values()
        for item in recommendations
    }

    if len(recommended_items) > n_items:
        raise ValueError("Recommended items exceed the catalogue size.")

    return len(recommended_items) / n_items


def gini_index(predicted: dict[str, list[int]]) -> float:

    item_counts = Counter(
        item
        for recommendations in predicted.values()
        for item in recommendations
    )

    if not item_counts:
        return 0.0

    frequencies = sorted(item_counts.values())
    n = len(frequencies)
    total = sum(frequencies)

    weighted_sum = sum(
        rank * count
        for rank, count in enumerate(frequencies, start=1)
    )

    return (2 * weighted_sum) / (n * total) - (n + 1) / n



def intra_list_diversity(
    predicted: dict[str, list[int]],
    articles: pd.DataFrame,
    attribute: str = "product_group_name"
) -> float:

    if "article_id" not in articles.columns or attribute not in articles.columns:
        raise ValueError("Articles must contain article_id and the selected attribute.")

    if articles["article_id"].duplicated().any():
        raise ValueError("Article IDs must be unique.")

    item_groups = articles.set_index("article_id")[attribute].to_dict()

    scores = []

    for recommendations in predicted.values():
        if not recommendations:
            scores.append(0.0)
            continue

        groups = {
            item_groups[item]
            for item in recommendations
            if item in item_groups and pd.notna(item_groups[item])
        }

        scores.append(len(groups) / len(recommendations))

    return sum(scores) / len(scores) if scores else 0.0



def novelty(predicted: dict[str, list[int]], item_popularity: pd.Series) -> float:

    if item_popularity.empty:
        return 0.0

    counts = pd.to_numeric(item_popularity)

    if counts.isna().any() or (counts < 0).any():
        raise ValueError("Item popularity must contain valid non-negative counts.")

    total_purchases = counts.sum()

    if total_purchases <= 0:
        return 0.0

    denominator = total_purchases + len(counts) + 1

    scores = []

    for recommendations in predicted.values():
        for item in recommendations:
            count = counts.get(item, 0)
            probability = (count + 1) / denominator
            scores.append(-np.log2(probability))

    return float(np.mean(scores)) if scores else 0.0



def evaluate(
    model: Recommender,
    test: pd.DataFrame,
    segments: pd.Series | None = None,
    k: int = K,
    *,
    train: pd.DataFrame | None = None,
    articles: pd.DataFrame | None = None,
    customers: pd.DataFrame | None = None,
) -> pd.DataFrame:

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not {"customer_id", "article_id"}.issubset(test.columns) or test.empty:
        raise ValueError("Test data must contain customer_id and article_id rows.")

    if test[["customer_id", "article_id"]].isna().any().any():
        raise ValueError("Test customer and article IDs cannot be missing.")

    actual = {
        str(customer_id): set(map(int, items))
        for customer_id, items in test.groupby(
            "customer_id", sort=False, observed=True
        )["article_id"]
    }

    customer_ids = list(actual)
    predicted = model.recommend_all(customer_ids, k=k)

    if set(predicted) != set(customer_ids):
        raise ValueError("Model must return recommendations for every test customer.")

    if any(len(items) != k or len(set(items)) != k for items in predicted.values()):
        raise ValueError("Each customer must receive k distinct recommendations.")

    if segments is None and train is not None:
        from .data import assign_segments
        segments = assign_segments(train, pd.Series(customer_ids))

    segment_labels = None

    if segments is not None:
        if segments.index.is_unique and all(cid in segments.index for cid in customer_ids):
            segment_labels = segments.loc[customer_ids].astype("string").tolist()
        elif len(segments) == len(customer_ids):
            segment_labels = segments.astype("string").tolist()
        else:
            raise ValueError("Segments must align with the test customer IDs.")

    n_items = articles["article_id"].nunique() if articles is not None else None

    popularity = (
        train.groupby("article_id", observed=True).size()
        if train is not None else None
    )

    groups = {"all": customer_ids}

    if segment_labels is not None:
        for label in ("new", "light", "regular"):
            groups[label] = [
                cid
                for cid, group in zip(customer_ids, segment_labels)
                if group == label
            ]

    if customers is not None:
        if not {"customer_id", "age"}.issubset(customers.columns):
            raise ValueError("Customer data must contain customer_id and age.")

        age_lookup = customers.drop_duplicates(
            "customer_id"
        ).set_index("customer_id")["age"]

        ages = pd.to_numeric(
            pd.Series(customer_ids).map(age_lookup),
            errors="coerce"
        )

        age_labels = pd.cut(
            ages,
            bins=[0, 24, 34, 44, 54, 64, float("inf")],
            labels=["0-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        ).astype("string").fillna("unknown").tolist()

        for label in dict.fromkeys(age_labels):
            groups[f"age:{label}"] = [
                cid
                for cid, group in zip(customer_ids, age_labels)
                if group == label
            ]

    rows = []

    for name, ids in groups.items():
        relevant = {cid: actual[cid] for cid in ids}
        recommendations = {cid: predicted[cid] for cid in ids}
        present = bool(ids)

        rows.append({
            "group": name,
            "n_customers": len(ids),
            f"MAP@{k}": map_at_k(relevant, recommendations, k) if present else np.nan,
            f"Recall@{k}": recall_at_k(relevant, recommendations, k) if present else np.nan,
            f"NDCG@{k}": ndcg_at_k(relevant, recommendations, k) if present else np.nan,
            f"HitRate@{k}": hit_rate_at_k(relevant, recommendations, k) if present else np.nan,
            "coverage": catalog_coverage(recommendations, n_items) if present and n_items else np.nan,
            "gini": gini_index(recommendations) if present else np.nan,
            "diversity": intra_list_diversity(recommendations, articles) if present and articles is not None else np.nan,
            "novelty": novelty(recommendations, popularity) if present and popularity is not None else np.nan,
        })

    return pd.DataFrame(rows).set_index("group")