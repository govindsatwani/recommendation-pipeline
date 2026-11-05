# Custom feature layer
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, path_for
from src.features.definitions import BY_ENTITY, FEATURES

DAY = 86400.0

def _mode_by_user(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby(["user_id", col]).agg(n=("rating", "size"), s=("rating", "sum")).reset_index()
    g = g.sort_values(["user_id", "n", "s"], ascending=[True, False, False])
    return g.drop_duplicates("user_id").set_index("user_id")[col]

def build_user_features(train: pd.DataFrame, items: pd.DataFrame, as_of: float) -> pd.DataFrame:
    t = train.merge(items[["item_id", "category", "brand", "price"]], on="item_id", how="left")
    g = t.groupby("user_id")
    out = pd.DataFrame(
        {
            "user_avg_rating": g["rating"].mean(),
            "user_rating_std": g["rating"].std().fillna(0.0),
            "user_activity_count": g.size(),
            "user_verified_ratio": g["verified"].mean(),
            "user_days_since_last_interaction": (as_of - g["timestamp"].max()) / DAY,
            "user_avg_price": g["price"].mean(),
        }
    )
    out["user_preferred_category"] = _mode_by_user(t, "category")
    out["user_preferred_brand"] = _mode_by_user(t, "brand")
    return out.reset_index()

def build_item_features(train: pd.DataFrame, items: pd.DataFrame, as_of: float) -> pd.DataFrame:
    g = train.groupby("item_id")
    stats = pd.DataFrame(
        {
            "item_popularity": g.size(),
            "item_unique_users": g["user_id"].nunique(),
            "item_avg_rating": g["rating"].mean(),
            "item_days_since_last_interaction": (as_of - g["timestamp"].max()) / DAY,
        }
    ).reset_index()
    out = items[["item_id", "price", "category", "subcategory", "brand"]].rename(
        columns={"price": "item_price", "category": "item_category", "subcategory": "item_subcategory", "brand": "item_brand"}
    )
    out = out.merge(stats, on="item_id", how="left")
    out["item_popularity"] = out["item_popularity"].fillna(0).astype(int)
    out["item_unique_users"] = out["item_unique_users"].fillna(0).astype(int)
    out["item_avg_rating"] = out["item_avg_rating"].fillna(train["rating"].mean())
    out["item_popularity_rank"] = out["item_popularity"].rank(method="min", ascending=False).astype(int)
    return out

def interaction_features(
    pairs: pd.DataFrame, user_feats: pd.DataFrame, item_feats: pd.DataFrame, as_of: float
) -> pd.DataFrame:
    df = pairs[["user_id", "item_id"] + [c for c in ("rating", "timestamp") if c in pairs.columns]].copy()
    df = df.merge(user_feats, on="user_id", how="left").merge(item_feats, on="item_id", how="left")
    out = df[["user_id", "item_id"]].copy()
    out["interaction_category_match"] = (df["item_category"] == df["user_preferred_category"]).astype(int)
    out["interaction_brand_match"] = (df["item_brand"] == df["user_preferred_brand"]).astype(int)
    out["interaction_price_ratio"] = df["item_price"] / df["user_avg_price"].replace(0, np.nan)
    if "rating" in df:
        out["interaction_rating_minus_user_avg"] = df["rating"] - df["user_avg_rating"]
    if "timestamp" in df:
        out["interaction_days_before_as_of"] = (as_of - df["timestamp"]) / DAY
    return out

class FeatureStore:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.dir = path_for(self.cfg, "features")
        self.log = get_logger("features", self.cfg)
        self._user: pd.DataFrame | None = None
        self._item: pd.DataFrame | None = None
        self.as_of: float | None = None

    # offline write
    def materialize(self, train: pd.DataFrame, items: pd.DataFrame, as_of: float) -> dict:
        user = build_user_features(train, items, as_of)
        item = build_item_features(train, items, as_of)
        inter = interaction_features(train, user, item, as_of)
        user.to_parquet(self.dir / "user_features.parquet", index=False)
        item.to_parquet(self.dir / "item_features.parquet", index=False)
        inter.to_parquet(self.dir / "interaction_features.parquet", index=False)
        meta = {
            "as_of_timestamp": float(as_of),
            "as_of_date": str(pd.to_datetime(as_of, unit="s", utc=True).date()),
            "n_user_rows": len(user),
            "n_item_rows": len(item),
            "n_interaction_rows": len(inter),
            "n_features": len(FEATURES),
        }
        (self.dir / "feature_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._user, self._item, self.as_of = user, item, float(as_of)
        self.log.info("Materialized %d features: users=%d items=%d interactions=%d", len(FEATURES), len(user), len(item), len(inter))
        return meta

    # online / read
    def load(self) -> "FeatureStore":
        self._user = pd.read_parquet(self.dir / "user_features.parquet")
        self._item = pd.read_parquet(self.dir / "item_features.parquet")
        self.as_of = json.loads((self.dir / "feature_metadata.json").read_text())["as_of_timestamp"]
        return self

    def _ensure(self):
        if self._user is None:
            self.load()

    def get_user_features(self, user_ids) -> pd.DataFrame:
        self._ensure()
        return pd.DataFrame({"user_id": list(user_ids)}).merge(self._user, on="user_id", how="left")

    def get_item_features(self, item_ids) -> pd.DataFrame:
        self._ensure()
        return pd.DataFrame({"item_id": list(item_ids)}).merge(self._item, on="item_id", how="left")

    def get_interaction_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        self._ensure()
        return interaction_features(pairs, self._user, self._item, self.as_of)

    @property
    def all_items(self) -> pd.DataFrame:
        self._ensure()
        return self._item

    @property
    def all_users(self) -> pd.DataFrame:
        self._ensure()
        return self._user

# automatically creating reports
def write_catalog(cfg: dict, store: FeatureStore) -> Path:
    rep = path_for(cfg, "reports")
    lines = [
        "# Feature Catalog",
        "",
        f"All features are computed from **training data only**",
        "",
    ]
    for entity, title in (("user", "User features"), ("item", "Item features"), ("interaction", "Interaction features")):
        lines += [f"## {title}", "", "| Feature | Type | Description | Formula / logic | Source |", "|---|---|---|---|---|"]
        for f in BY_ENTITY[entity]:
            lines.append(f"| `{f.name}` | {f.dtype} | {f.description} | `{f.formula}` | {f.source} |")
        lines.append("")
    path = rep / "feature_catalog.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    proc = path_for(cfg, "processed")
    train = pd.read_parquet(proc / "train.parquet")
    items = pd.read_parquet(proc / "items_clean.parquet")
    info = json.loads((proc / "split_info.json").read_text())
    store = FeatureStore(cfg)
    meta = store.materialize(train, items, as_of=info["cutoff_timestamp"])
    write_catalog(cfg, store)
    return meta

if __name__ == "__main__":
    run()
