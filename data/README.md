# Data

The data comes from the Kaggle competition
**[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)**.

It is **not stored in this repository**: competition data may not be redistributed, and
`transactions_train.csv` (~3 GB) is far above GitHub's 100 MB file limit.

## Each member, once

1. Create a Kaggle account and open the competition page above.
2. Go to the **Rules** tab and click **I Understand and Accept**. Without this the download fails with a 403 error.
3. Create an API token: Kaggle → Settings → API → **Create New Token**. This downloads `kaggle.json`.
4. Move it to `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
   On Mac/Linux run `chmod 600 ~/.kaggle/kaggle.json`. **Never commit this file.**
5. From the repository root run:

```bash
bash scripts/download_data.sh
```

This downloads only the three CSV files (no product images) into `data/raw/`.

## What goes where

| Folder | Contents | In git? |
|---|---|---|
| `data/raw/` | `transactions_train.csv`, `customers.csv`, `articles.csv` | No |
| `data/processed/` | shared train/test split made by Member 1 (parquet) | No |

Member 1 shares the code that builds `data/processed/`, so everyone can recreate the same split.

## Manual alternative

Download the three CSVs from the competition's **Data** tab in the browser and put them in `data/raw/`.
