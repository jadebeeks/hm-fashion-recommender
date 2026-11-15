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

    transaction_dtypes = {
        "customer_id": "category",
        "article_id": "int32",
        "price": "float32",
        "sales_channel_id": "int8",
    }

    transactions = pd.read_csv(
        RAW_DIR / "transactions_train.csv",
        dtype=transaction_dtypes,
        parse_dates=["t_dat"],
        nrows=nrows,
    )

    customer_dtypes = {
        "customer_id": "category",
        "FN": "float32",
        "Active": "float32",
        "club_member_status": "category",
        "fashion_news_frequency": "category",
        "age": "Int16",
        "postal_code": "category",
    }

    customers = pd.read_csv(
        RAW_DIR / "customers.csv",
        dtype=customer_dtypes,
        nrows=nrows,
    )

    article_dtypes = {
        "article_id": "int32",
        "product_type_name": "category",
        "product_group_name": "category",
        "graphical_appearance_name": "category",
        "colour_group_name": "category",
        "department_name": "category",
        "section_name": "category",
        "garment_group_name": "category",
    }

    articles = pd.read_csv(
        RAW_DIR / "articles.csv",
        dtype=article_dtypes,
        nrows=nrows,
    )

    return transactions, customers, articles


def make_subset(transactions: pd.DataFrame, train_weeks: int = 12) -> pd.DataFrame:

    if train_weeks < 1:
        raise ValueError("train_weeks must be at least 1.")

    if "t_dat" not in transactions.columns:
        raise ValueError("Transactions must contain a 't_dat' column.")

    if transactions.empty:
        return transactions.copy()

    dates = pd.to_datetime(transactions["t_dat"])

    end_date = dates.max().normalize() + pd.Timedelta(days=1)

    start_date = end_date - pd.Timedelta(weeks=train_weeks + 1)

    mask = (dates >= start_date) & (dates < end_date)

    return transactions.loc[mask].copy()


def temporal_split(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    if transactions.empty:
        raise ValueError("Cannot split an empty dataset.")

    if "t_dat" not in transactions.columns:
        raise ValueError("Transactions must contain a 't_dat' column.")

    dates = pd.to_datetime(transactions["t_dat"])

    if dates.isna().any():
        raise ValueError("Transaction dates cannot contain missing values.")

    end_date = dates.max().normalize() + pd.Timedelta(days=1)
    test_start = end_date - pd.Timedelta(days=7)

    train = transactions.loc[dates < test_start].copy().reset_index(drop=True)
    test = transactions.loc[(dates >= test_start) & (dates < end_date)].copy().reset_index(drop=True)

    if train.empty or test.empty:
        raise ValueError("Both training and testing datasets must contain transactions.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    return train, test


def assign_segments(train: pd.DataFrame, customer_ids: pd.Series) -> pd.Series:

    if "customer_id" not in train.columns:
        raise ValueError("Training data must contain a 'customer_id' column.")

    purchase_counts = train.groupby("customer_id", observed=True).size()

    counts = customer_ids.astype("string").map(purchase_counts).fillna(0).astype("int64")

    segments = pd.cut(
        counts,
        bins=[-1, 0, 4, float("inf")],
        labels=["new", "light", "regular"],
    )

    return segments.rename("segment")


def new_items(train: pd.DataFrame, test: pd.DataFrame) -> set[int]:

    if "article_id" not in train.columns or "article_id" not in test.columns:
        raise ValueError("Both datasets must contain an 'article_id' column.")

    train_items = set(map(int, train["article_id"].dropna().unique()))
    test_items = set(map(int, test["article_id"].dropna().unique()))

    return test_items - train_items


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:

    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Training or testing data not found. Run temporal_split() first."
        )

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    return train, test