# Demo of live recommendations
from __future__ import annotations

import pandas as pd

from src.common import get_logger, load_config, path_for
from src.inference.recommend import RecommendationService

def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)

def run(cfg: dict | None = None, k: int = 5) -> str:
    cfg = cfg or load_config()
    log = get_logger("demo", cfg)
    proc, rep = path_for(cfg, "processed"), path_for(cfg, "reports")
    svc = {m: RecommendationService(cfg, m) for m in ("svd", "popularity")}
    train = pd.read_parquet(proc / "train.parquet")
    test = pd.read_parquet(proc / "test.parquet")

    liked = set(test.loc[test["rating"] >= cfg["evaluation"]["relevance_threshold"], "user_id"]) & set(train["user_id"])
    counts = train[train["user_id"].isin(liked)].groupby("user_id").size()
    counts = counts[(counts >= 6) & (counts <= 12)].sort_index()
    user = counts.index[len(counts) // 2]
    future = test[(test["user_id"] == user) & (test["rating"] >= cfg["evaluation"]["relevance_threshold"])].merge(
        svc["svd"].data.items[["item_id", "title"]], on="item_id")

    pop_items = train["item_id"].value_counts().index[[20, 30]]
    history = {pop_items[0]: 5.0, pop_items[1]: 4.0}
    titles = svc["svd"].data.items.set_index("item_id")["title"]

    out = ["# Sample Recommendations (live prediction demo)", "",
           f"## 1. Known user `{user}`", "", "**Past reviews (training period):**", "",
           _table(svc["svd"].user_history(user).assign(event_time=lambda d: d["event_time"].dt.date)), ""]
    for m in ("svd", "popularity"):
        out += [f"**Top-{k} - {m}**", "", _table(svc[m].recommend(user, k).drop(columns=["item_id"])), ""]
    if len(future):
        out += ["**Items this user actually rated >= 4 in the later test period:**", "",
                _table(future[["item_id", "title", "rating"]].head(10)), ""]
    out += ["## 2. New user described only by reviews", "",
            "Reviews supplied: " + "; ".join(f"{titles.get(i, i)[:50]} -> {r:.0f}*" for i, r in history.items()), "",
            _table(svc["svd"].recommend(None, k, history).drop(columns=["item_id"])), "",
            "## 3. Cold start", "",
            _table(svc["svd"].recommend("UNKNOWN_USER", k).drop(columns=["item_id"])), ""]
    path = rep / "sample_recommendations.md"
    path.write_text("\n".join(out), encoding="utf-8")
    log.info("Wrote %s", path.name)
    return str(path)

if __name__ == "__main__":
    run()
