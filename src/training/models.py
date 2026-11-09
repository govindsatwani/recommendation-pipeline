# Recommendation models
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, normalize

from src.common import load_config, path_for
from src.features.feature_store import FeatureStore

RATING_MIN, RATING_MAX = 1.0, 5.0

@dataclass
class TrainData:
    R: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray
    store: FeatureStore
    items: pd.DataFrame
    seed: int = 42
    user_idx: dict = field(init=False)
    item_idx: dict = field(init=False)

    def __post_init__(self):
        self.user_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_idx = {i: j for j, i in enumerate(self.item_ids)}

    @property
    def global_mean(self) -> float:
        return float(self.R.data.mean())

    def row_from_history(self, history: dict[str, float]) -> sp.csr_matrix:
        cols = [self.item_idx[i] for i in history if i in self.item_idx]
        vals = [float(history[i]) for i in history if i in self.item_idx]
        return sp.csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, len(self.item_ids)))

def load_training_data(cfg: dict | None = None) -> TrainData:
    import scipy.sparse as _sp

    cfg = cfg or load_config()
    proc = path_for(cfg, "processed")
    users = pd.read_parquet(proc / "user_index.parquet")
    items = pd.read_parquet(proc / "item_index.parquet")
    return TrainData(
        R=_sp.load_npz(proc / "user_item_matrix.npz").tocsr(),
        user_ids=users["user_id"].to_numpy(),
        item_ids=items["item_id"].to_numpy(),
        store=FeatureStore(cfg).load(),
        items=pd.read_parquet(proc / "items_clean.parquet"),
        seed=cfg["project"]["seed"],
    )

def item_attribute_matrix(data: TrainData) -> sp.csr_matrix:
    f = data.store.get_item_features(data.item_ids)
    attrs = f[["item_category", "item_subcategory", "item_brand"]].fillna("none").astype(str)
    return normalize(OneHotEncoder(handle_unknown="ignore").fit_transform(attrs)).tocsr()

# models
class Recommender:
    name = "base"

    def fit(self, data: TrainData, cfg: dict) -> "Recommender":
        raise NotImplementedError

    def score(self, R_rows: sp.csr_matrix) -> np.ndarray:
        raise NotImplementedError

    def predict_rating(self, users, items) -> np.ndarray | None:
        return None

    def get_params(self) -> dict:
        return {}

class _RatingBaseline:

    def __init__(self, data: TrainData, prior: float):
        self.mu = data.global_mean
        u = data.store.get_user_features(data.user_ids)
        i = data.store.get_item_features(data.item_ids)
        n_u, n_i = u["user_activity_count"].to_numpy(float), i["item_popularity"].to_numpy(float)
        self.b_u = n_u / (n_u + prior) * (u["user_avg_rating"].to_numpy() - self.mu)
        self.b_i = n_i / (n_i + prior) * (i["item_avg_rating"].to_numpy() - self.mu)
        self.user_idx, self.item_idx = data.user_idx, data.item_idx

    def predict(self, users, items, use_user: bool = True) -> np.ndarray:
        ui = np.array([self.user_idx.get(u, -1) for u in users])
        ii = np.array([self.item_idx.get(i, -1) for i in items])
        bu = np.where(ui >= 0, self.b_u[np.clip(ui, 0, None)], 0.0) if use_user else 0.0
        bi = np.where(ii >= 0, self.b_i[np.clip(ii, 0, None)], 0.0)
        return self.mu + bu + bi

class PopularityRecommender(Recommender):
    # Baseline: recommend the most-interacted-with items
    name = "popularity"

    def __init__(self, prior_strength: float = 5.0):
        self.prior = prior_strength

    def fit(self, data, cfg):
        self.prior = cfg["models"]["popularity"]["prior_strength"]
        feats = data.store.get_item_features(data.item_ids)
        self.pop = feats["item_popularity"].to_numpy(float)
        self.baseline = _RatingBaseline(data, self.prior)
        return self

    def score(self, R_rows):
        return np.tile(self.pop, (R_rows.shape[0], 1))

    def predict_rating(self, users, items):
        return np.clip(self.baseline.predict(users, items, use_user=False), RATING_MIN, RATING_MAX)

    def get_params(self):
        return {"ranking": "item_popularity (interaction count)", "rating_prior_strength": self.prior}

class SVDRecommender(Recommender):
    # Matrix factorisation (truncated SVD)
    name = "svd"

    def __init__(self, n_factors: int = 20, rating_factors: int = 20):
        self.n_factors, self.rating_factors = n_factors, rating_factors

    def fit(self, data, cfg):
        seed = data.seed
        k = min(self.n_factors, min(data.R.shape) - 1)
        self.V = TruncatedSVD(n_components=k, random_state=seed).fit(data.R).components_
        self.baseline = _RatingBaseline(data, cfg["models"]["popularity"]["prior_strength"])
        coo = data.R.tocoo()
        pred = self.baseline.mu + self.baseline.b_u[coo.row] + self.baseline.b_i[coo.col]
        E = sp.csr_matrix((coo.data - pred, (coo.row, coo.col)), shape=data.R.shape)
        rk = min(self.rating_factors, min(data.R.shape) - 1)
        svd = TruncatedSVD(n_components=rk, random_state=seed)
        self.U_res = svd.fit_transform(E)
        self.V_res = svd.components_
        self.user_idx, self.item_idx = data.user_idx, data.item_idx
        return self

    def score(self, R_rows):
        return np.asarray((R_rows @ self.V.T) @ self.V)

    def predict_rating(self, users, items):
        base = self.baseline.predict(users, items)
        ui = np.array([self.user_idx.get(u, -1) for u in users])
        ii = np.array([self.item_idx.get(i, -1) for i in items])
        ok = (ui >= 0) & (ii >= 0)
        res = np.zeros(len(users))
        res[ok] = np.einsum("ij,ji->i", self.U_res[ui[ok]], self.V_res[:, ii[ok]])
        return np.clip(base + res, RATING_MIN, RATING_MAX)

    def get_params(self):
        return {"n_factors": self.n_factors, "rating_factors": self.rating_factors, "algorithm": "TruncatedSVD"}

# ranking
def top_k(scores: np.ndarray, R_rows: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    s = scores.astype(float).copy()
    seen = R_rows.tocoo()
    s[seen.row, seen.col] = -np.inf
    k = min(k, s.shape[1])
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(s, part, axis=1).argsort(axis=1)[:, ::-1]
    idx = np.take_along_axis(part, order, axis=1)
    return idx, np.take_along_axis(s, idx, axis=1)
