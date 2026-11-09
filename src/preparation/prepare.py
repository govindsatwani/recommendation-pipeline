# Data preparation
from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd

from src.common import get_logger, load_config, path_for

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
BRAND_PREFIX_RE = re.compile(r"^by\s+", re.IGNORECASE)
PRICE_RE = re.compile(r"^\$?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?)$")
MAX_LEVEL_LEN = 50 

def clean_text(value) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return None
    text = TAG_RE.sub(" ", html.unescape(str(value)))
    text = WS_RE.sub(" ", text).strip()
    return text or None

def parse_price(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    m = PRICE_RE.match(str(value).strip())
    return float(m.group(1).replace(",", "")) if m else np.nan

def clean_brand(value) -> str | None:
    text = clean_text(value)
    return BRAND_PREFIX_RE.sub("", text) or None if text else None

def split_categories(path) -> tuple[str, str | None]:
    if path is None or pd.isna(path):
        return "Uncategorized", None
    levels = [clean_text(p) for p in str(path).split(" | ")]
    levels = [p for p in levels if p and len(p) <= MAX_LEVEL_LEN]
    if levels and levels[0].lower() == "software":
        levels = levels[1:]
    cat = levels[0] if levels else "Uncategorized"
    sub = levels[1] if len(levels) > 1 else None
    return cat, sub

def prepare_items(items: pd.DataFrame) -> pd.DataFrame:
    df = items.copy()
    df["title"] = df["title"].map(clean_text).fillna("Unknown title")
    df["brand"] = df["brand"].map(clean_brand).fillna("Unknown")
    df["main_category"] = df["main_category"].map(clean_text).fillna("Unknown")
    cats = df["categories"].map(split_categories)
    df["category"] = [c[0] for c in cats]
    df["subcategory"] = [c[1] for c in cats]
    df["description"] = df["description"].map(clean_text).fillna("").str.slice(0, 1000)
    df["price"] = df["price"].map(parse_price)
    df["price_missing"] = df["price"].isna()
    cat_median = df.groupby("category")["price"].transform("median") # first imputing the median of the item's category, then the global median
    df["price"] = df["price"].fillna(cat_median).fillna(df["price"].median()).round(2)
    return df.drop(columns=["categories"]).drop_duplicates("item_id").reset_index(drop=True)

def prepare_users(users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    df["user_name"] = df["user_name"].map(clean_text).fillna("Anonymous")
    return df.drop_duplicates("user_id").reset_index(drop=True)

def prepare_interactions(inter: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = inter.copy()
    stats = {"rows_in": int(len(df))}
    df["user_id"] = df["user_id"].str.strip()
    df["item_id"] = df["item_id"].str.strip()
    df["summary"] = df["summary"].map(clean_text).fillna("")
    df["helpful_votes"] = df["vote"].fillna(0).astype(int)
    df["event_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["rating"] = df["rating"].astype(float)
    df = df[df["user_id"].isin(users["user_id"]) & df["item_id"].isin(items["item_id"])]
    stats["rows_out"] = int(len(df))
    cols = ["user_id", "item_id", "rating", "timestamp", "event_time", "verified", "helpful_votes", "summary"]
    return df[cols].sort_values("timestamp", kind="stable").reset_index(drop=True), stats

def time_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    cutoff = int(np.ceil(np.quantile(df["timestamp"].to_numpy(), 1 - test_fraction)))

    train = (
        df[df["timestamp"] < cutoff]
        .sort_values("timestamp", kind="stable")
        .drop_duplicates(["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )
    test = (
        df[df["timestamp"] >= cutoff]
        .sort_values("timestamp", kind="stable")
        .drop_duplicates(["user_id", "item_id"], keep="last")
        .reset_index(drop=True)
    )
    return train, test, cutoff

def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("preparation", cfg)
    src, out = path_for(cfg, "validated"), path_for(cfg, "processed")

    users = prepare_users(pd.read_parquet(src / "users.parquet"))
    items = prepare_items(pd.read_parquet(src / "items.parquet"))
        inter, stats = prepare_interactions(pd.read_parquet(src / "interactions.parquet"), users, items)

    train, test, cutoff = time_split(inter, cfg["split"]["test_fraction"])
    stats["repeated_user_item_pairs_removed"] = len(inter) - len(train) - len(test)
    log.info(
        "Duplicates resolved within time partitions: %d repeated user-item rows removed",
        stats["repeated_user_item_pairs_removed"],
    )
    assert train["timestamp"].max() < test["timestamp"].min(), "time split leaked"

    train_users, train_items = set(train["user_id"]), set(train["item_id"])
    info = {
        **stats,
        "cutoff_timestamp": cutoff,
        "cutoff_date": str(pd.to_datetime(cutoff, unit="s", utc=True).date()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_fraction_actual": round(len(test) / (len(train) + len(test)), 4),
        "train_users": len(train_users),
        "train_items": len(train_items),
        "test_rows_cold_user": int((~test["user_id"].isin(train_users)).sum()),
        "test_rows_cold_item": int((~test["item_id"].isin(train_items)).sum()),
        "train_period": [str(train["event_time"].min().date()), str(train["event_time"].max().date())],
        "test_period": [str(test["event_time"].min().date()), str(test["event_time"].max().date())],
    }

    users.to_parquet(out / "users_clean.parquet", index=False)
    items.to_parquet(out / "items_clean.parquet", index=False)
    inter.to_parquet(out / "interactions_clean.parquet", index=False)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    with open(out / "split_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)
    log.info("Split at %s: train=%d test=%d (cold users in test: %d rows)", info["cutoff_date"], len(train), len(test), info["test_rows_cold_user"])
    return info

if __name__ == "__main__":
    run()
