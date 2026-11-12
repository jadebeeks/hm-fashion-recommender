# Right Customer, Right Item: H&M fashion recommender

Group project for **Recommendation Engines** (Bachelor in Data and Business Analytics, IE University, 2026–27).

**Scenario:** We are the data science team at a fashion retailer that currently shows products in random order. We design, build and evaluate a recommender that does better, and present its business value to a non-technical CEO.

**Research question:** Compared with random display and a "trending now" baseline, how much more of next week's purchasing can a hybrid recommender capture across new, light and regular customers? What does it trade off in diversity and catalogue coverage?

## Team

See **[ROLES.md](ROLES.md)** for each member's full responsibilities.

| # | Role | Branch |
|---|------|--------|
| 1 | Data & Evaluation Lead | `m1-data-eval` |
| 2 | Baselines & New Customers | `m2-baselines` |
| 3 | Collaborative Filtering | `m3-collaborative` |
| 4 | Content-Based & New Products | `m4-content` |
| 5 | Hybrid, Business Value & Presentation Lead | `m5-hybrid-business` |

## How the system works

```
                     ┌─ new customer (0 purchases)   → trends for their age band      (Member 2)
recommend(customer) ─┼─ light customer (1–4)          → content-based + trends         (Members 4 + 2)
                     └─ regular customer (5+)         → collaborative filtering/hybrid (Members 3 + 5)
new products (no sales yet)                           → content similarity             (Member 4)
```

All models share one interface (`src/hm_recsys/base.py`) and one scoring function (`src/hm_recsys/evaluation.py`).

## Repository layout

```
├── data/
│   ├── raw/                  Kaggle CSVs go here (not in git)
│   └── processed/            shared train/test split (not in git)
├── docs/                     data dictionary, A/B test plan, architecture, ethics
├── notebooks/
│   ├── 00_main_pipeline.ipynb               end-to-end run (Members 1 + 5)
│   ├── 01_eda.ipynb                         Member 1
│   ├── 02_baselines.ipynb                   Member 2
│   ├── 03_collaborative_filtering.ipynb     Member 3
│   ├── 04_content_based.ipynb               Member 4
│   └── 05_hybrid_and_business_value.ipynb   Member 5
├── reports/
│   ├── figures/              charts for the slides
│   └── presentation/         the CEO deck
├── scripts/download_data.sh  downloads the dataset from Kaggle
├── src/hm_recsys/            reusable code, one module per role
└── tests/                    quick checks for the metrics
```

## Getting started

1. Read **[data/README.md](data/README.md)**. You need a Kaggle account and must accept the competition rules.
2. Follow the setup in **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Dataset

[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) on Kaggle: transactions, customers and articles (product metadata).

**The data is not stored in this repository.** Kaggle competition data may not be redistributed, and the transactions file (~3 GB) is too large for GitHub. Each member downloads it with `scripts/download_data.sh`.

## AI use

Generative AI was used to help plan the project and set up this repository. Each member keeps a short log of their own AI use for the final acknowledgement the syllabus asks for.
