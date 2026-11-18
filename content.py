"""Content-based recommender and the new-item path.

Owner: Member 4 (Content-Based & New Products), branch ``m4-content``.

How it works
------------
1. **Item text.** Each article becomes one short text: product name, attributes
   (type, group, pattern, colour, department, section...) and the written
   description (``build_item_text``).
2. **Item vectors.** The texts are turned into vectors with Bag of Words or
   TF-IDF (optionally after lemmatisation), or any embedding passed in
   ``item_vectors`` (e.g. sentence-BERT). Vectors are L2-normalised, so a dot
   product is a cosine similarity.
3. **Customer profile.** The recency-weighted mean of the vectors of everything
   the customer bought. A purchase ``d`` days before the end of training gets
   weight ``0.5 ** (d / half_life)``, so with the default 14 days a purchase from
   two weeks ago counts half as much as one from yesterday.
4. **Scoring.** Cosine similarity between the profile and every *candidate*
   item. Candidates are the items that sold in the last ``candidate_weeks``
   weeks (a proxy for "still in the shop") plus any launch items we pass in.
   Optionally the score gets a bonus for items that are selling right now
   (``popularity_weight``), which turns the model into "content + trends".
5. **Fallback.** Customers without purchases get a fallback model (trending
   now by default), so every customer gets ``k`` distinct items.

New items
---------
A product that has never been sold has no purchase data, so collaborative
filtering and popularity cannot rank it. It does have a description, so this
model can score it from day one:

- ``ContentBasedRecommender(candidate_items=launch, only_candidate_items=True)``
  recommends launch items only (one row of "new in for you");
- ``audience(article_ids, n)`` goes the other way: for each new item, the
  customers whose profile matches it best (who to email about a launch).
- ``AttributePopularityRecommender`` is the no-model baseline for launches:
  push the launches whose product type sells best right now.

Optional: ``sentence_embeddings`` turns the item texts into sentence-BERT
vectors (ONNX, CPU) to pass as ``item_vectors``.

Quick use
---------
>>> model = ContentBasedRecommender().fit(train, articles)
>>> model.recommend("some-customer-id")        # always 12 distinct article ids
>>> model.explain("some-customer-id", 706016001)  # "because you bought ..."
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import Recommender
from .baselines import PopularityRecommender, _last_weeks, _top_up

# Attribute columns of articles.csv that describe what the product *is*.
# Codes and numeric ids are left out: they duplicate the names.
ATTRIBUTE_COLUMNS = (
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "perceived_colour_master_name",
    "department_name",
    "index_name",
    "index_group_name",
    "section_name",
    "garment_group_name",
)
TEXT_FIELDS = ("name", "attributes", "description")
_MISSING = {"", "nan", "none", "unknown", "undefined", "-1"}


# ---------------------------------------------------------------------------
# 1. Item text
# ---------------------------------------------------------------------------
def _clean(col: pd.Series) -> pd.Series:
    s = col.astype("string").fillna("").str.strip()
    return s.where(~s.str.lower().isin(_MISSING), "")


def build_item_text(articles: pd.DataFrame,
                    fields: Sequence[str] = TEXT_FIELDS) -> pd.Series:
    """One text per article: product name, attributes and description.

    ``fields`` picks the parts: any of ``"name"``, ``"attributes"``,
    ``"description"``. Placeholder values such as "Unknown" are dropped, so
    they don't make unrelated products look alike.

    Returns a Series of strings indexed by ``article_id``.
    """
    unknown = set(fields) - set(TEXT_FIELDS)
    if unknown:
        raise ValueError(f"unknown text fields {unknown}; use {TEXT_FIELDS}")
    if "article_id" not in articles.columns:
        raise ValueError("articles must have an article_id column")

    parts = []
    if "name" in fields and "prod_name" in articles.columns:
        parts.append(_clean(articles["prod_name"]))
    if "attributes" in fields:
        cols = [c for c in ATTRIBUTE_COLUMNS if c in articles.columns]
        if cols:
            attrs = pd.concat([_clean(articles[c]) for c in cols], axis=1)
            parts.append(attrs.apply(lambda r: " ".join(v for v in r if v), axis=1))
    if "description" in fields and "detail_desc" in articles.columns:
        parts.append(_clean(articles["detail_desc"]))

    if parts:
        table = pd.concat(parts, axis=1)
        text = table.apply(lambda r: ". ".join(v for v in r if v), axis=1)
    else:
        text = pd.Series("", index=articles.index)
    return pd.Series(text.to_numpy(dtype=object),
                     index=articles["article_id"].astype("int64").to_numpy(),
                     name="text")


_LEMMA_CACHE: dict[str, str] = {}


def lemmatize(texts: pd.Series, model: str = "en_core_web_sm",
              batch_size: int = 500) -> pd.Series:
    """Lower-cased lemmas of each text ("pockets" -> "pocket", "covered" -> "cover").

    Needs spaCy and its small English model:
    ``pip install spacy && python -m spacy download en_core_web_sm``.
    Many articles share a description (colour variants), so each distinct
    text is processed once, and results are cached for the session.
    """
    todo = [t for t in pd.unique(texts.to_numpy(dtype=object)) if t not in _LEMMA_CACHE]
    if todo:
        try:
            import spacy
        except ImportError as err:  # pragma: no cover - optional dependency
            raise ImportError("lemmatize() needs spaCy: pip install spacy && "
                              "python -m spacy download en_core_web_sm") from err
        nlp = spacy.load(model, disable=["parser", "ner"])
        for text, doc in zip(todo, nlp.pipe(todo, batch_size=batch_size)):
            _LEMMA_CACHE[text] = " ".join(t.lemma_.lower() for t in doc if t.is_alpha)
    return texts.map(_LEMMA_CACHE)


def sentence_embeddings(texts: pd.Series, model_dir, batch_size: int = 128,
                        max_length: int = 128) -> pd.DataFrame:
    """Sentence-BERT embeddings (mean-pooled, L2-normalised) from an ONNX export.

    ``model_dir`` must hold ``model.onnx`` and ``tokenizer.json`` of a
    sentence-transformers model, e.g. all-MiniLM-L6-v2 (the Xenova ONNX export
    on Hugging Face). ONNX runs on CPU without PyTorch. Each distinct text is
    embedded once. Returns a DataFrame indexed like ``texts``, usable as
    ``ContentBasedRecommender(item_vectors=...)``.

    The tokenizer is the lower-cased BERT WordPiece vocabulary; if only
    ``vocab.txt`` is at hand, ``tokenizers.BertWordPieceTokenizer(vocab,
    lowercase=True).save("tokenizer.json")`` rebuilds it.
    """
    from pathlib import Path

    import onnxruntime as ort
    from tokenizers import Tokenizer

    model_dir = Path(model_dir)
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_truncation(max_length)
    tok.enable_padding()
    sess = ort.InferenceSession(str(model_dir / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    inputs = {i.name for i in sess.get_inputs()}

    unique = pd.unique(texts.fillna("").to_numpy(dtype=object))
    # similar lengths in a batch -> less padding -> faster
    order = np.argsort([len(t) for t in unique], kind="stable")
    chunks = []
    for s in range(0, len(unique), batch_size):
        enc = tok.encode_batch([unique[i] for i in order[s:s + batch_size]])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run(None, feed)[0]                       # batch x tokens x dim
        m = mask[..., None].astype(np.float32)
        chunks.append((hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None))
    out = np.empty((len(unique), chunks[0].shape[1]), dtype=np.float32)
    out[order] = normalize(np.vstack(chunks))
    row = pd.Index(unique).get_indexer(texts.fillna("").to_numpy(dtype=object))
    return pd.DataFrame(out[row], index=texts.index)


def vectorize(texts: pd.Series, method: str = "tfidf", min_df: int = 2,
              ngram_range: tuple[int, int] = (1, 1)):
    """Bag of Words or TF-IDF matrix (one L2-normalised row per text) and the fitted vectorizer.

    English stop words are removed and words in fewer than ``min_df`` articles
    are dropped (typos and one-off names).
    """
    if method == "bow":
        vec = CountVectorizer(stop_words="english", min_df=min_df, ngram_range=ngram_range)
    elif method == "tfidf":
        vec = TfidfVectorizer(stop_words="english", min_df=min_df, ngram_range=ngram_range,
                              sublinear_tf=True)
    else:
        raise ValueError("method must be 'bow' or 'tfidf'")
    X = vec.fit_transform(texts.fillna(""))
    return normalize(X.astype(np.float32)), vec


def _dense(m) -> np.ndarray:
    return m.toarray() if sp.issparse(m) else np.asarray(m)


# ---------------------------------------------------------------------------
# 2. The recommender
# ---------------------------------------------------------------------------
class ContentBasedRecommender(Recommender):
    """Recommends products that look like what the customer bought, recent purchases first.

    Parameters
    ----------
    vectorizer : ``"tfidf"`` (default) or ``"bow"``. Ignored if ``item_vectors`` is given.
    recency_half_life_days : purchases this many days old count half. ``None`` = all equal.
    lemmatize : lemmatise the item text before vectorising (needs spaCy).
    text_fields : which parts of the item text to use (see ``build_item_text``).
    item_vectors : optional DataFrame (index ``article_id``) of precomputed
        embeddings, e.g. sentence-BERT. Replaces BoW/TF-IDF.
    candidate_weeks : only recommend items sold in the last ``candidate_weeks``
        weeks of train ("still in the shop"). ``None`` = the whole catalogue.
    candidate_items : extra article ids that are always candidates (launches).
    only_candidate_items : recommend *only* ``candidate_items`` (the new-item path).
    popularity_weight : bonus for items selling in the last ``popularity_weeks``
        weeks, added to the cosine score. 0 = pure content.
    exclude_purchased : never recommend an item the customer already bought.
    one_per_product : at most one colour variant of each ``product_code`` per list.
    fallback : model for customers with no purchases (default: trending now).
    """

    def __init__(self, vectorizer: str = "tfidf", recency_half_life_days: float | None = 14.0,
                 *, lemmatize: bool = False, text_fields: Sequence[str] = TEXT_FIELDS,
                 item_vectors: pd.DataFrame | None = None,
                 candidate_weeks: int | None = 2, candidate_items: Iterable[int] | None = None,
                 only_candidate_items: bool = False,
                 popularity_weight: float = 0.0, popularity_weeks: int = 1,
                 exclude_purchased: bool = True, one_per_product: bool = False,
                 fallback: Recommender | None = None, min_df: int = 2,
                 batch_size: int = 1000):
        self.vectorizer = vectorizer
        self.recency_half_life_days = recency_half_life_days
        self.lemmatize = lemmatize
        self.text_fields = tuple(text_fields)
        self.item_vectors = item_vectors
        self.candidate_weeks = candidate_weeks
        self.candidate_items = None if candidate_items is None else list(candidate_items)
        self.only_candidate_items = only_candidate_items
        self.popularity_weight = popularity_weight
        self.popularity_weeks = popularity_weeks
        self.exclude_purchased = exclude_purchased
        self.one_per_product = one_per_product
        self.fallback = fallback
        self.min_df = min_df
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        base = "content_vectors" if self.item_vectors is not None else f"content_{self.vectorizer}"
        if self.lemmatize and self.item_vectors is None:
            base += "_lemma"
        if self.only_candidate_items:
            base += "_launches"
        if self.popularity_weight:
            base += "+trends"
        return base

    # -- fitting -------------------------------------------------------------
    def fit(self, train, articles=None, customers=None):
        if articles is None:
            raise ValueError("ContentBasedRecommender needs the articles table")
        if self.only_candidate_items and not self.candidate_items:
            raise ValueError("only_candidate_items=True needs candidate_items")

        articles = articles.drop_duplicates("article_id")
        self.profile_norms_ = None
        self._fit_items(articles)
        train = train[train["article_id"].isin(self.item_index_)]
        self._fit_profiles(train)
        self._fit_popularity(train)
        self._fit_candidates(train)

        if self.fallback is None:
            self.fallback = PopularityRecommender(weeks=self.popularity_weeks)
        self.fallback.fit(train, articles, customers)
        self.fallback_list_ = [int(i) for i in self.fallback.recommend("__unknown__", k=500)]
        return self

    def _fit_items(self, articles: pd.DataFrame) -> None:
        if self.item_vectors is not None:
            vecs = self.item_vectors
            vecs = vecs[~vecs.index.duplicated()]
            ids = vecs.index.astype("int64")
            X = normalize(np.asarray(vecs, dtype=np.float32))
            self.texts_ = None
        else:
            texts = build_item_text(articles, self.text_fields)
            if self.lemmatize:
                texts = lemmatize(texts)
            self.texts_ = texts
            X, self.vectorizer_ = vectorize(texts, self.vectorizer, self.min_df)
            ids = texts.index
        self.item_index_ = pd.Index(np.asarray(ids, dtype=np.int64))
        self.X_ = X.tocsr() if sp.issparse(X) else X
        codes = (articles.set_index("article_id")["product_code"]
                 if "product_code" in articles.columns else None)
        self.product_code_ = (codes.reindex(self.item_index_).to_numpy()
                              if codes is not None else self.item_index_.to_numpy())

    def _fit_profiles(self, train: pd.DataFrame) -> None:
        dates = pd.to_datetime(train["t_dat"])
        self.end_date_ = dates.max()
        days = (self.end_date_ - dates).dt.days.to_numpy(dtype=np.float32)
        hl = self.recency_half_life_days
        weights = np.ones_like(days) if not hl else np.power(0.5, days / float(hl))

        cust = train["customer_id"].astype(str)
        self.customer_index_ = pd.Index(pd.unique(cust.to_numpy()))
        rows = self.customer_index_.get_indexer(cust)
        cols = self.item_index_.get_indexer(train["article_id"].astype("int64"))
        W = sp.csr_matrix((weights, (rows, cols)),
                          shape=(len(self.customer_index_), len(self.item_index_)),
                          dtype=np.float32)
        W.sum_duplicates()
        self.W_ = W

    def _fit_popularity(self, train: pd.DataFrame) -> None:
        recent = _last_weeks(train, self.popularity_weeks)
        buyers = recent.groupby("article_id", observed=True)["customer_id"].nunique()
        pop = np.zeros(len(self.item_index_), dtype=np.float32)
        pos = self.item_index_.get_indexer(buyers.index.astype("int64"))
        pop[pos] = np.log1p(buyers.to_numpy(dtype=np.float32))
        self.popularity_ = pop / pop.max() if pop.max() > 0 else pop
        # "what is selling now", as a content profile: used for customers with
        # no history in the launch-only mode
        trend = self.popularity_
        self.trend_profile_ = normalize(
            sp.csr_matrix(trend[None, :]) @ self.X_ if sp.issparse(self.X_)
            else trend[None, :] @ self.X_)

    def _fit_candidates(self, train: pd.DataFrame) -> None:
        if self.only_candidate_items:
            pool = pd.Index(self.candidate_items, dtype="int64")
        else:
            if self.candidate_weeks is None:
                pool = self.item_index_
            else:
                pool = pd.Index(_last_weeks(train, self.candidate_weeks)["article_id"]
                                .astype("int64").unique())
            if self.candidate_items:
                pool = pool.union(pd.Index(self.candidate_items, dtype="int64"))
        pos = self.item_index_.get_indexer(pool)
        pos = np.unique(pos[pos >= 0])
        if len(pos) == 0:
            raise ValueError("no candidate items with a text/vector")
        self.pool_pos_ = pos
        self.pool_ids_ = self.item_index_.to_numpy()[pos]
        self.C_ = self.X_[pos]
        self.col_to_pool_ = np.full(len(self.item_index_), -1, dtype=np.int64)
        self.col_to_pool_[pos] = np.arange(len(pos))

    # -- scoring -------------------------------------------------------------
    def _profiles(self, rows: np.ndarray):
        """L2-normalised recency-weighted profiles for customer rows."""
        P = self.W_[rows] @ self.X_
        return normalize(P) if sp.issparse(P) else normalize(np.asarray(P))

    def _scores(self, rows: np.ndarray) -> np.ndarray:
        S = _dense(self._profiles(rows) @ self.C_.T).astype(np.float32, copy=False)
        if self.popularity_weight:
            S += self.popularity_weight * self.popularity_[self.pool_pos_][None, :]
        if self.exclude_purchased:
            bought = self.W_[rows]
            r, c = bought.nonzero()
            p = self.col_to_pool_[c]
            keep = p >= 0
            S[r[keep], p[keep]] = -np.inf
        return S

    def _top_from_scores(self, S: np.ndarray, k: int) -> list[list[int]]:
        n = S.shape[1]
        m = min(n, k * 4 if self.one_per_product else k)
        top = np.argpartition(-S, m - 1, axis=1)[:, :m] if m < n else np.tile(np.arange(n), (len(S), 1))
        order = np.take_along_axis(S, top, axis=1).argsort(axis=1)[:, ::-1]
        top = np.take_along_axis(top, order, axis=1)
        out = []
        for i, row in enumerate(top):
            items, seen_codes = [], set()
            for p in row:
                if not np.isfinite(S[i, p]):
                    break
                if self.one_per_product:
                    code = self.product_code_[self.pool_pos_[p]]
                    if code in seen_codes:
                        continue
                    seen_codes.add(code)
                items.append(int(self.pool_ids_[p]))
                if len(items) == k:
                    break
            out.append(items)
        return out

    def _cold_lists(self, k: int) -> list[int]:
        """Launch-only mode, no history: launches closest to what is selling now."""
        s = _dense(self.trend_profile_ @ self.C_.T).ravel()
        return self._top_from_scores(s[None, :], k)[0]

    def recommend_all(self, customer_ids, k=12):
        ids = [str(c) for c in customer_ids]
        rows = self.customer_index_.get_indexer(ids)
        out: dict[str, list[int]] = {}

        cold = [c for c, r in zip(ids, rows) if r < 0]
        if cold:
            if self.only_candidate_items:
                shared = _top_up(self._cold_lists(k), self.fallback_list_, k)
                out.update({c: list(shared) for c in cold})
            else:
                out.update({c: list(v) for c, v in
                            self.fallback.recommend_all(cold, k=k).items()})

        warm = [(c, r) for c, r in zip(ids, rows) if r >= 0]
        for start in range(0, len(warm), self.batch_size):
            chunk = warm[start:start + self.batch_size]
            lists = self._top_from_scores(self._scores(np.array([r for _, r in chunk])), k)
            for (c, r), items in zip(chunk, lists):
                if len(items) < k:
                    items = self._top_up_unbought(items, r, k)
                out[c] = items
        return {c: out[c] for c in ids}

    def _top_up_unbought(self, items: list[int], row: int, k: int) -> list[int]:
        bought = set()
        if self.exclude_purchased:
            bought = set(self.item_index_.to_numpy()[self.W_[row].indices].tolist())
        pool = [i for i in self.fallback_list_ if i not in bought]
        out = _top_up(items, pool, k)
        if len(out) < k:  # tiny catalogues (tests): allow anything
            out = _top_up(out, self.fallback_list_ + self.pool_ids_.tolist(), k)
        return out

    def recommend(self, customer_id, k=12):
        return self.recommend_all([customer_id], k)[str(customer_id)]

    # -- extras for the notebook, the hybrid and the demo ------------------------
    def history(self, customer_id) -> pd.DataFrame:
        """What the customer bought and how much each purchase weighs in the profile."""
        r = self.customer_index_.get_indexer([str(customer_id)])[0]
        if r < 0:
            return pd.DataFrame(columns=["article_id", "weight"])
        row = self.W_[r]
        return (pd.DataFrame({"article_id": self.item_index_.to_numpy()[row.indices],
                              "weight": row.data})
                  .sort_values("weight", ascending=False, ignore_index=True))

    def explain(self, customer_id, article_id: int, top: int = 3) -> pd.DataFrame:
        """"Because you bought...": the past purchases most similar to ``article_id``."""
        hist = self.history(customer_id)
        if hist.empty:
            return hist.assign(similarity=[])
        pos = self.item_index_.get_indexer([int(article_id)])[0]
        if pos < 0:
            raise KeyError(article_id)
        target = self.X_[[pos]]
        past = self.X_[self.item_index_.get_indexer(hist["article_id"])]
        sims = _dense(past @ target.T).ravel()
        return (hist.assign(similarity=sims)
                    .sort_values("similarity", ascending=False, ignore_index=True)
                    .head(top))

    def similar_items(self, article_id: int, k: int = 12,
                      within: Iterable[int] | None = None) -> pd.Series:
        """"More like this": the ``k`` items most similar to one article (cosine)."""
        pos = self.item_index_.get_indexer([int(article_id)])[0]
        if pos < 0:
            raise KeyError(article_id)
        cand = (np.arange(len(self.item_index_)) if within is None
                else self.item_index_.get_indexer(pd.Index(list(within), dtype="int64")))
        cand = cand[(cand >= 0) & (cand != pos)]
        sims = _dense(self.X_[cand] @ self.X_[pos].T).ravel()
        best = np.argsort(-sims)[:k]
        return pd.Series(sims[best], index=self.item_index_.to_numpy()[cand[best]],
                         name="similarity")

    def profile_norms(self) -> np.ndarray:
        """Length of every customer's (unnormalised) profile; cached. Used by ``audience``."""
        if self.profile_norms_ is None:
            norms = np.empty(self.W_.shape[0], dtype=np.float32)
            step = 20_000
            for s in range(0, self.W_.shape[0], step):
                P = self.W_[s:s + step] @ self.X_
                norms[s:s + step] = (np.sqrt(P.multiply(P).sum(axis=1)).A1 if sp.issparse(P)
                                     else np.linalg.norm(P, axis=1))
            norms[norms == 0] = 1.0
            self.profile_norms_ = norms
        return self.profile_norms_

    def audience(self, article_ids: Iterable[int], n: int = 1000,
                 universe: Iterable[str] | None = None,
                 batch: int = 50) -> dict[int, list[str]]:
        """For each article, the ``n`` customers whose profile matches it best.

        This is the new-item path seen from marketing: who should hear about a
        launch. Only customers with purchase history can be ranked;
        ``universe`` limits the choice further (e.g. recently active customers).
        """
        ids = [int(a) for a in article_ids]
        pos = self.item_index_.get_indexer(pd.Index(ids, dtype="int64"))
        if (pos < 0).any():
            raise KeyError([a for a, p in zip(ids, pos) if p < 0])
        norms = self.profile_norms()
        allowed = None
        if universe is not None:
            rows = self.customer_index_.get_indexer(pd.Index([str(c) for c in universe]))
            allowed = np.zeros(len(self.customer_index_), dtype=bool)
            allowed[rows[rows >= 0]] = True
        n = min(n, len(self.customer_index_) if allowed is None else int(allowed.sum()))
        cust = self.customer_index_.to_numpy()
        out = {}
        for s in range(0, len(ids), batch):
            sims_items = _dense(self.X_ @ self.X_[pos[s:s + batch]].T)  # items x batch
            scores = np.asarray(self.W_ @ sims_items) / norms[:, None]  # customers x batch
            if allowed is not None:
                scores[~allowed] = -np.inf
            top = np.argpartition(-scores, n - 1, axis=0)[:n]
            for j in range(scores.shape[1]):
                col = top[:, j]
                col = col[np.argsort(-scores[col, j])]
                out[ids[s + j]] = cust[col].tolist()
        return out


# ---------------------------------------------------------------------------
# 3. A no-personal-data baseline for launches
# ---------------------------------------------------------------------------
class AttributePopularityRecommender(Recommender):
    """Ranks candidate items by how well their *attribute* is selling right now.

    A brand-new item has no sales, but its product type (or department, colour...)
    does. This is what a merchandiser would do without any model: push the
    launches in the categories that sell. With ``spread=True`` the list takes
    one launch per category in turn (best-selling category first), so twelve
    slots are not all trousers; launches inside a category are in a fixed
    random order, since nothing else distinguishes them. Everyone gets the same
    list, so this is the baseline a personalised launch row has to beat.
    """
    name = "attribute_popularity"

    def __init__(self, candidate_items: Iterable[int], attribute: str = "product_type_name",
                 weeks: int = 1, spread: bool = True, seed: int = 42):
        self.candidate_items = list(candidate_items)
        self.attribute = attribute
        self.weeks = weeks
        self.spread = spread
        self.seed = seed

    def fit(self, train, articles=None, customers=None):
        if articles is None:
            raise ValueError("AttributePopularityRecommender needs the articles table")
        attr = articles.drop_duplicates("article_id").set_index("article_id")[self.attribute]
        recent = _last_weeks(train, self.weeks)
        sales = (recent.assign(a=recent["article_id"].map(attr))
                       .groupby("a", observed=True)["customer_id"].nunique())
        cand = pd.DataFrame({"article_id": pd.Index(self.candidate_items, dtype="int64").unique()})
        cand["attr"] = cand["article_id"].map(attr)
        cand["score"] = cand["attr"].map(sales).astype(float).fillna(0)
        cand["tie"] = np.random.default_rng(self.seed).permutation(len(cand))
        cand = cand.sort_values(["score", "tie"], ascending=[False, True])
        if self.spread:
            # round-robin: 1st launch of each category, then the 2nd of each, ...
            cand["turn"] = cand.groupby("attr", sort=False, dropna=False).cumcount()
            cand = cand.sort_values(["turn", "score", "tie"], ascending=[True, False, True])
        self.ranking_ = _top_up(cand["article_id"].astype(int).tolist(),
                                PopularityRecommender(self.weeks).fit(train).ranking_, 500)
        return self

    def recommend(self, customer_id, k=12):
        return self.ranking_[:k]

    def recommend_all(self, customer_ids, k=12):
        return {c: list(self.ranking_[:k]) for c in customer_ids}
