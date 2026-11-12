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

## Timeline

Internal team deadlines, all **end of day (23:59)**. **Working system by Tuesday 17 November; everything finished by Friday 20 November.** The weekend of 21–22 November is buffer only.
The official course deadline still needs confirming with the professor (Member 5). If it's earlier, we compress; if later, we keep this plan and use the extra time for polish.

| Day | Date | Milestone | Who |
|-----|------|-----------|-----|
| Thu | 12 Nov | Accept the GitHub invite, set up Python, download the data (`data/README.md`). | Everyone |
| Fri | 13 Nov | **Split + `evaluate()` merged into `dev`.** The whole team depends on this. | Member 1 |
| Fri | 13 Nov | Everyone else: first version of their method working on a small sample of the data. | Members 2–5 |
| Sat–Sun | 14–15 Nov | Build and score your models on the real split. Member 5: hybrid routing, business-value table, A/B test plan, deck outline. | Everyone |
| Mon | 16 Nov | **All models scored with `evaluate()`; pull requests into `dev` opened and reviewed.** | Members 2, 3, 4 |
| Tue | 17 Nov | **Working system: `00_main_pipeline.ipynb` runs end to end, hybrid scored, results table complete.** | Members 1 + 5 (all on call) |
| Wed | 18 Nov | Interpretation: charts in `reports/figures/`, markdown explanations in every notebook, business-value numbers final. | Everyone |
| Thu | 19 Nov | **Slides complete** (each member writes their own section), first full rehearsal. | Everyone; Member 5 edits |
| Fri | 20 Nov | **Final rehearsal, AI-use acknowledgement written, `dev` merged into `main`. Done.** | Everyone |

### Per-member deadlines

| Member | Must be in `dev` by | What |
|--------|---------------------|------|
| 1 | **Fri 13 Nov** | Split files code, segments, new-item list, `evaluate()` with all metrics |
| 1 | Wed 18 Nov | `01_eda.ipynb` finished with explanations; data dictionary |
| 2 | Mon 16 Nov | Five baselines scored; new-customer path |
| 3 | Mon 16 Nov | Item-to-item CF and ALS scored by segment; failure analysis |
| 4 | Mon 16 Nov | TF-IDF model scored; new-item path; diversity analysis |
| 5 | Tue 17 Nov | Hybrid scored; `00_main_pipeline.ipynb` runs end to end (with Member 1) |
| 5 | Wed 18 Nov | Business-value scenarios, A/B test plan, architecture and ethics notes, CEO demo |
| All | Thu 19 Nov | Own slides in the deck |

### If someone falls behind

- Say so in the group chat **as soon as** a deadline looks at risk, not on the day.
- Cut extensions first: sentence-BERT, the weighted hybrid and the bandit sketch are optional.
- If Member 1 is late on Friday, everyone keeps going on a sample of the data, so no one sits idle.
