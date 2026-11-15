import pandas as pd
import hm_recsys.data as data


def test_load_raw(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()

    pd.DataFrame({
        "t_dat": ["2020-09-15", "2020-09-16"],
        "customer_id": ["C1", "C2"],
        "article_id": [101, 102],
        "price": [0.1, 0.2],
        "sales_channel_id": [1, 2]
    }).to_csv(raw / "transactions_train.csv", index=False)

    pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "age": [22, 37]
    }).to_csv(raw / "customers.csv", index=False)

    pd.DataFrame({
        "article_id": [101, 102],
        "product_group_name": ["Tops", "Shoes"]
    }).to_csv(raw / "articles.csv", index=False)

    monkeypatch.setattr(data, "RAW_DIR", raw)

    transactions, customers, articles = data.load_raw()

    assert len(transactions) == 2
    assert len(customers) == 2
    assert len(articles) == 2
    assert pd.api.types.is_datetime64_any_dtype(transactions["t_dat"])
    assert str(transactions["article_id"].dtype) == "int32"


def test_make_subset():
    df = pd.DataFrame({
        "t_dat": pd.date_range("2020-06-01", "2020-09-22"),
        "article_id": 101
    })

    subset = data.make_subset(df)

    assert len(subset) == 91
    assert subset["t_dat"].min() == pd.Timestamp("2020-06-24")
    assert subset["t_dat"].max() == pd.Timestamp("2020-09-22")


def test_temporal_split_and_load(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "t_dat": pd.date_range("2020-06-24", "2020-09-22"),
        "customer_id": ["C1"] * 91,
        "article_id": [101] * 91
    })

    monkeypatch.setattr(data, "PROCESSED_DIR", tmp_path)

    train, test = data.temporal_split(df)
    loaded_train, loaded_test = data.load_split()

    assert len(train) == 84
    assert len(test) == 7
    assert train["t_dat"].max() < test["t_dat"].min()

    pd.testing.assert_frame_equal(train, loaded_train)
    pd.testing.assert_frame_equal(test, loaded_test)


def test_assign_segments():
    train = pd.DataFrame({
        "customer_id": ["C1"] * 3 + ["C2"] * 5
    })

    ids = pd.Series(["C1", "C2", "C3"])
    segments = data.assign_segments(train, ids)

    assert segments.astype("string").tolist() == [
        "light", "regular", "new"
    ]


def test_new_items():
    train = pd.DataFrame({
        "article_id": [101, 102, 103]
    })

    test = pd.DataFrame({
        "article_id": [101, 104, 105]
    })

    assert data.new_items(train, test) == {104, 105}
