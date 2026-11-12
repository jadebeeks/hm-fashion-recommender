# Team roles

Five members, five clear areas. Each person owns specific files, works on their own branch, and presents their own part. Fill in the names below once roles are agreed.

| # | Role | Name | Branch | Main files owned |
|---|------|------|--------|------------------|
| 1 | Data & Evaluation Lead | _____ | `m1-data-eval` | `src/hm_recsys/data.py`, `src/hm_recsys/evaluation.py`, `notebooks/01_eda.ipynb` |
| 2 | Baselines & New Customers | _____ | `m2-baselines` | `src/hm_recsys/baselines.py`, `notebooks/02_baselines.ipynb` |
| 3 | Collaborative Filtering | _____ | `m3-collaborative` | `src/hm_recsys/collaborative.py`, `notebooks/03_collaborative_filtering.ipynb` |
| 4 | Content-Based & New Products | _____ | `m4-content` | `src/hm_recsys/content.py`, `notebooks/04_content_based.ipynb` |
| 5 | Hybrid, Business Value & Presentation Lead | _____ | `m5-hybrid-business` | `src/hm_recsys/hybrid.py`, `src/hm_recsys/business.py`, `notebooks/05_hybrid_and_business_value.ipynb`, `docs/` |

Shared by everyone: `src/hm_recsys/base.py` (the common interface; change it only after the team agrees) and `notebooks/00_main_pipeline.ipynb` (owned by Members 1 and 5 together).

---

## Member 1: Data & Evaluation Lead

**Goal:** give the whole team clean data, one fair train/test split and one scoring function. Everyone else depends on this, so it comes first.

**Tasks**
1. Download the data (`scripts/download_data.sh`) and load it with memory-saving data types.
2. Explore the data in `01_eda.ipynb`:
   - sales over time and by season
   - customer ages
   - the long tail of rarely bought items
   - how often customers re-buy
   - online vs store sales channel
3. Choose the working subset (suggested: last 12 weeks for training, the following week for testing) and justify it.
4. Build the time-based split and save it to `data/processed/` so everyone uses the same files.
5. Define customer segments by number of past purchases: **new** (0), **light** (1–4), **regular** (5+). Also list the **new items** that appear only in the test week.
6. Implement `evaluate()`:
   - accuracy: MAP@12, Recall@12, NDCG@12, hit rate
   - catalogue coverage
   - Gini (how concentrated recommendations are on a few items)
   - diversity within each list
   - novelty
   - results broken down by segment and by age band
7. Check for leakage: no test-week information can be used in training features.
8. Keep `docs/data_dictionary.md` up to date.

**Done when:** the split files and `evaluate()` are merged into `dev` and the other four can score a model with one line.
**Presents:** "Our data and how we tested fairly".

---

## Member 2: Baselines & New Customers

**Goal:** show what random display costs the company, and build the strategy for customers we know nothing about.

**Tasks**
1. Build these recommenders, each using the shared interface in `base.py`:
   - Random: today's situation
   - Global popularity
   - Recent popularity: best sellers of the last 1–2 weeks ("trending now")
   - Popularity by age band (demographic filtering)
   - "Buy again": the customer's own most-bought items
2. Build the **new-customer path**: age-band trends when age is known, global trends when it isn't.
3. Analyse popularity bias: how concentrated are popular-item recommendations, and which items never get shown?
4. Write up why "trending now" is the real baseline to beat, not random.

**Done when:** all five baselines are scored with `evaluate()` and the new-customer path is used by the hybrid.
**Presents:** "Where the company stands today, and quick wins".

---

## Member 3: Collaborative Filtering

**Goal:** recommend based on what similar customers bought.

**Tasks**
1. Item-to-item collaborative filtering: "customers who bought X also bought Y", based on items bought by the same customers.
2. Matrix factorisation with ALS on purchase data (the `implicit` library). Only light tuning: a small grid over the number of factors and the regularisation.
3. Compare the two methods by segment, and show **where collaborative filtering fails** (light customers, new items). This tells the hybrid when not to use it.
4. Provide a fallback for customers or items the model has never seen.

**Done when:** both models are scored by segment and the failure analysis is written up.
**Presents:** "Learning from similar customers, and where that stops working".

---

## Member 4: Content-Based & New Products

**Goal:** recommend by what the product is, so new items and light customers are covered.

**Tasks**
1. Build an item text from the description plus attributes (product type, colour, pattern, department). Compare Bag-of-Words with TF-IDF, and optionally try lemmatisation.
2. Build customer profiles from the items they bought, weighting recent purchases more.
3. Build the **new-item path**: new products are recommended to customers whose profiles are similar.
4. Analyse diversity and novelty: are content-based lists too similar, for example all black T-shirts?
5. *Optional extension:* sentence-BERT embeddings, compared with TF-IDF.

**Done when:** the content model is scored overall, on light customers and on new items.
**Presents:** "Recommending by product features, including brand-new items".

---

## Member 5: Hybrid, Business Value & Presentation Lead

**Goal:** turn the parts into one system that serves every customer, and prove its value to the CEO.

**Tasks**
1. **Switching hybrid:**
   - new customers → Member 2's path
   - light customers → content-based plus trends
   - regular customers → collaborative filtering
   Then try a weighted blend for regular customers.
2. Write the final `recommend(customer_id)` that works for **any** customer ID, including unknown ones.
3. **Business value model:**
   - Convert hit rates into extra purchases and revenue relative to random and "trending now".
   - Use scenarios, and label every assumption. Prices in the dataset appear to be scaled, so check this.
4. **A/B test plan** (`docs/ab_test_plan.md`): what to randomise, main metric, guardrail metrics, rough sample size.
5. **Bandit sketch:** how to give new items some exposure (explore) while showing proven items (exploit).
6. Architecture, monitoring and ethics notes in `docs/`.
7. **CEO demo:** three personas (new customer, light shopper, loyal customer), with random vs our recommendations side by side.
8. Lead the slide deck: structure, story and consistency. Everyone writes their own slides.

**Done when:** `00_main_pipeline.ipynb` runs end to end and the deck is complete.
**Presents:** the opening (problem), the demo, the business value and the recommendation to the CEO.

---

## Shared responsibilities

- **Code review rotation:** each pull request into `dev` needs one approval.

  | Author | Reviewer |
  |--------|----------|
  | Member 1 | Member 2 |
  | Member 2 | Member 3 |
  | Member 3 | Member 4 |
  | Member 4 | Member 5 |
  | Member 5 | Member 1 |

- **Everyone** writes the markdown explanations in their own notebook (why this method, what the results mean) and their own slides.
- **Everyone** uses `evaluate()` from Member 1. No private scoring code.
- **AI use:** keep a short log of how you used AI tools. The syllabus requires an acknowledgement in the submission.

## Member by member

| Member | Builds | Deadlines |
|--------|--------|-----------|
| **1: Data & Evaluation** | Cleans the data, splits it into training and test data, builds the scoring tool | **Fri 13:** split + scoring tool · **Wed 18:** exploration notebook |
| **2: Baselines** | Simple recommenders (random, best-sellers, best-sellers by age) and the plan for new customers | **Mon 16:** all done and scored |
| **3: Collaborative Filtering** | "Customers who bought this also bought…" models | **Mon 16:** all done and scored |
| **4: Content-Based** | Recommends by product features such as type and colour, and handles brand-new products | **Mon 16:** all done and scored |
| **5: Hybrid & Business** | Combines all models into one system, calculates the business value, leads the presentation | **Tue 17:** system runs end to end · **Wed 18:** business value, A/B plan, demo |
| **Everyone** | Their own slides | **Thu 19:** slides · **Fri 20:** final rehearsal |
