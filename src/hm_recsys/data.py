"""Loading, subsetting, splitting and segmenting the H&M data.

Owner: Member 1 (Data & Evaluation Lead), branch ``m1-data-eval``.
Everyone imports from here, so this is the first module to finish.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_raw(nrows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load transactions, customers and articles with memory-saving dtypes.

    TODO (Member 1):
    - parse ``t_dat`` as a date;
    - use category / int32 / float32 dtypes where possible;
    - return (transactions, customers, articles).
    """
    raise NotImplementedError


def make_subset(transactions: pd.DataFrame, train_weeks: int = 12) -> pd.DataFrame:
    """Keep the last ``train_weeks`` weeks plus the final test week.

    TODO (Member 1): justify the window size in 01_eda.ipynb.
    """
    raise NotImplementedError


def temporal_split(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: everything before the final week is train, the final week is test.

    Never split randomly; that leaks future trends into training.
    TODO (Member 1): save both parts to data/processed/ as parquet.
    """
    raise NotImplementedError


def assign_segments(train: pd.DataFrame, customer_ids: pd.Series) -> pd.Series:
    """Label each customer as 'new' (0), 'light' (1-4) or 'regular' (5+ purchases in train).

    TODO (Member 1)
    """
    raise NotImplementedError


def new_items(train: pd.DataFrame, test: pd.DataFrame) -> set[int]:
    """Articles that appear in the test week but never in train (new-item cold start).

    TODO (Member 1)
    """
    raise NotImplementedError


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the frozen train/test split from data/processed/ (what everyone else uses).

    TODO (Member 1)
    """
    raise NotImplementedError
