"""Translate offline results into business value for the CEO.

Owner: Member 5 (Hybrid, Business Value & Presentation Lead), branch ``m5-hybrid-business``.
"""
from __future__ import annotations

import pandas as pd


def purchases_per_1000_recs(results: pd.DataFrame) -> pd.DataFrame:
    """Turn hit rates into "purchases caught per 1,000 recommendations" per model.

    TODO (Member 5)
    """
    raise NotImplementedError


def revenue_scenarios(results: pd.DataFrame, assumptions: dict) -> pd.DataFrame:
    """Scenario table of extra revenue vs random and vs 'trending now'.

    Every assumption (customers reached, conversion, average price) must be
    passed in explicitly and shown on the slide. Prices in the dataset appear scaled.
    TODO (Member 5)
    """
    raise NotImplementedError
