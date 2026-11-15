import pandas as pd

from hm_recsys.data import RAW_DIR, make_subset, temporal_split

start_date = "2020-06-24"
end_date = "2020-09-22"

transaction_dtypes = {
    "t_dat": "string",
    "customer_id": "string",
    "article_id": "int32",
    "price": "float32",
    "sales_channel_id": "int8",
}

chunks = []

for chunk in pd.read_csv(
    RAW_DIR / "transactions_train.csv",
    dtype=transaction_dtypes,
    chunksize=250_000,
):
    filtered = chunk.loc[
        chunk["t_dat"].between(start_date, end_date)
    ]

    if not filtered.empty:
        chunks.append(filtered.copy())

if not chunks:
    raise ValueError("No transactions found in the selected period.")

transactions = pd.concat(chunks, ignore_index=True)

transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
transactions["customer_id"] = transactions["customer_id"].astype("category")

subset = make_subset(transactions, train_weeks=12)

if len(subset) != 3_904_391:
    raise ValueError("The selected transaction count is unexpected.")

train, test = temporal_split(subset)

assert len(train) == 3_664_080
assert len(test) == 240_311
assert train["t_dat"].max() < test["t_dat"].min()

print("Training transactions:", len(train))
print("Testing transactions:", len(test))
print("Training period:", train["t_dat"].min(), "to", train["t_dat"].max())
print("Testing period:", test["t_dat"].min(), "to", test["t_dat"].max())
print("Data preparation completed successfully!")
