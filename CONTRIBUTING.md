# How we work together

## Branches

```
main                 final, presentable version (only updated from dev)
└── dev              integration: everyone's finished work comes together here
    ├── m1-data-eval
    ├── m2-baselines
    ├── m3-collaborative
    ├── m4-content
    └── m5-hybrid-business
```

- Work **only on your own branch**.
- Never push directly to `main` or `dev`. Use a pull request (PR).
- `dev` is merged into `main` at the end of each phase, once the main notebook runs.

## First-time setup

```bash
git clone https://github.com/jadebeeks/hm-fashion-recommender.git
cd hm-fashion-recommender
git checkout m1-data-eval          # use YOUR branch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash scripts/download_data.sh      # see data/README.md first
```

## Daily routine

```bash
git checkout <your-branch>
git pull origin dev                # get the team's latest work
# ... work ...
git add <files>
git commit -m "m3: add item-to-item CF"
git push origin <your-branch>
```

When a piece is finished, open a PR from your branch into `dev` on GitHub. Your reviewer is listed in `ROLES.md`.

## Rules

1. **Never commit data.** `data/raw/` and `data/processed/` are ignored by git. Competition data cannot be shared publicly, and the files are too large for GitHub.
2. **Never commit your Kaggle token** (`kaggle.json`).
3. **Use the shared interface.** Every model subclasses `Recommender` in `src/hm_recsys/base.py` and is scored with `evaluate()` from `src/hm_recsys/evaluation.py`.
4. **Put reusable code in `src/`, explanations in notebooks.** Notebooks import from `src/` so the main pipeline can reuse your work.
5. **Before committing a notebook:** Restart & Run All, so the outputs match the code. Don't save huge outputs, such as printed dataframes with thousands of rows.
6. **Edit only your own files.** If you need to change someone else's file or `base.py`, ask them first.
7. **Commit message format:** start with your member tag, for example `m2: add popularity by age band`.

## The shared interface (contract)

```python
from hm_recsys.base import Recommender

class MyModel(Recommender):
    name = "my_model"

    def fit(self, train, articles=None, customers=None):
        ...
        return self

    def recommend(self, customer_id, k=12):
        ...  # must return a list of k article_ids, for ANY customer_id (even unknown)
```

Scoring:

```python
from hm_recsys.evaluation import evaluate
results = evaluate(MyModel().fit(train, articles, customers), test, segments=segments)
```
