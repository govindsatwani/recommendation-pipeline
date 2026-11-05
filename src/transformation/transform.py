# Data transformation
from __future__ import annotations

import pandas as pd
import scipy.sparse as sp

from src.common import get_logger, load_config, path_for

def user_activity_summary(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = g.agg(
        n_interactions=("item_id", "size"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        first_interaction_ts=("timestamp", "min"),
        last_interaction_ts=("timestamp", "max"),
        verified_ratio=("verified", "mean"),
    ).reset_index()
    out["std_rating"] = out["std_rating"].fillna(0.0)
    out["active_days"] = ((out["last_interaction_ts"] - out["first_interaction_ts"]) / 86400).round(1)
    return out

def item_popularity(train: pd.DataFrame) -> pd.DataFrame:
    g = train.groupby("item_id")
    out = g.agg(
        n_interactions=("user_id", "size"),
        n_users=("user_id", "nunique"),
        avg_rating=("rating", "mean"),
        last_interaction_ts=("timestamp", "max"),
    ).reset_index()
    out["popularity_rank"] = out["n_interactions"].rank(method="min", ascending=False).astype(int)
    return out.sort_values("popularity_rank").reset_index(drop=True)

def build_matrix(train: pd.DataFrame) -> tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    users = pd.DataFrame({"user_id": sorted(train["user_id"].unique())})
    items = pd.DataFrame({"item_id": sorted(train["item_id"].unique())})
    users["user_idx"] = range(len(users))
    items["item_idx"] = range(len(items))
    rows = train["user_id"].map(users.set_index("user_id")["user_idx"])
    cols = train["item_id"].map(items.set_index("item_id")["item_idx"])
    mat = sp.csr_matrix((train["rating"].to_numpy(), (rows, cols)), shape=(len(users), len(items)))
    return mat, users, items

def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("transformation", cfg)
    proc = path_for(cfg, "processed")
    train = pd.read_parquet(proc / "train.parquet")

    uas, pop = user_activity_summary(train), item_popularity(train)
    mat, users, items = build_matrix(train)

    uas.to_parquet(proc / "user_activity_summary.parquet", index=False)
    pop.to_parquet(proc / "item_popularity.parquet", index=False)
    users.to_parquet(proc / "user_index.parquet", index=False)
    items.to_parquet(proc / "item_index.parquet", index=False)
    sp.save_npz(proc / "user_item_matrix.npz", mat)

    density = mat.nnz / (mat.shape[0] * mat.shape[1])
    summary = {"users": mat.shape[0], "items": mat.shape[1], "nnz": int(mat.nnz), "density": round(density, 5)}
    log.info("user_activity_summary=%d rows | item_popularity=%d rows | matrix %s density=%.4f", len(uas), len(pop), mat.shape, density)
    return summary

if __name__ == "__main__":
    run()
