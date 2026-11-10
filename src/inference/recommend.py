# Code for live recommendations
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.common import get_logger, load_config, path_for
from src.training.models import TrainData, load_training_data, top_k

MODELS = ("popularity", "svd")
DEFAULT_MODEL = "svd"

class RecommendationService:
    def __init__(self, cfg: dict | None = None, model: str = DEFAULT_MODEL):
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        self.cfg = cfg or load_config()
        self.log = get_logger("inference", self.cfg)
        self.data: TrainData = load_training_data(self.cfg)
        models_dir = path_for(self.cfg, "models")
        self.model_name = model
        self.model = joblib.load(models_dir / f"{model}.joblib")
        self.fallback = joblib.load(models_dir / "popularity.joblib")
        self.catalog = self.data.store.get_item_features(self.data.item_ids).merge(
            self.data.items[["item_id", "title"]], on="item_id", how="left"
        )

    def _present(self, idx: np.ndarray, scores: np.ndarray, source: str) -> pd.DataFrame:
        rows = self.catalog.iloc[idx]
        out = pd.DataFrame(
            {
                "rank": range(1, len(idx) + 1),
                "item_id": rows["item_id"].to_numpy(),
                "title": rows["title"].fillna("").str.slice(0, 70).to_numpy(),
                "brand": rows["item_brand"].to_numpy(),
                "category": rows["item_category"].to_numpy(),
                "price": rows["item_price"].to_numpy(),
                "score": np.round(scores, 4),
            }
        )
        out.attrs["source"] = source
        return out

    def user_history(self, user_id: str) -> pd.DataFrame:
        proc = path_for(self.cfg, "processed")
        train = pd.read_parquet(proc / "train.parquet")
        h = train[train["user_id"] == user_id].merge(self.data.items[["item_id", "title", "brand", "category"]], on="item_id")
        return h.sort_values("timestamp")[["event_time", "item_id", "title", "brand", "category", "rating"]]

    def recommend(self, user_id: str | None = None, k: int = 10, history: dict[str, float] | None = None) -> pd.DataFrame:
        if history is not None:
            R = self.data.row_from_history(history)
            idx, sc = top_k(self.model.score(R), R, k)
            return self._present(idx[0], sc[0], f"{self.model_name} (fold-in from {R.nnz} supplied reviews)")
        if user_id in self.data.user_idx:
            R = self.data.R[[self.data.user_idx[user_id]]]
            idx, sc = top_k(self.model.score(R), R, k)
            return self._present(idx[0], sc[0], f"{self.model_name} (known user)")
        self.log.warning("Unknown user without usable history -> popularity fallback")
        R = self.data.R[:1] * 0
        idx, sc = top_k(self.fallback.score(R), R, k)
        return self._present(idx[0], sc[0], "popularity (cold-start fallback)")

def recommend(user_id: str | None = None, k: int = 10, model: str = DEFAULT_MODEL, history: dict[str, float] | None = None) -> pd.DataFrame:
    return RecommendationService(model=model).recommend(user_id, k, history)
