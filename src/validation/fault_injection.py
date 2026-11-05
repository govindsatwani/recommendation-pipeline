# Fault injection -> forcefully adding wrong values in data to check the working of the validate phase
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import load_config, path_for

def inject_faults(users: pd.DataFrame, items: pd.DataFrame, inter: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)
    users, items, inter = (d.copy().astype(object) for d in (users, items, inter))
    n = len(inter)
    pick = lambda k: rng.choice(n, size=k, replace=False)

    inter.loc[inter.index[pick(5)], "rating"] = None
    inter.loc[inter.index[pick(5)], "rating"] = "7"
    inter.loc[inter.index[pick(5)], "rating"] = "0"
    inter.loc[inter.index[pick(5)], "rating"] = "five"
    inter.loc[inter.index[pick(5)], "timestamp"] = str(int(pd.Timestamp("2099-01-01").timestamp()))
    inter.loc[inter.index[pick(5)], "timestamp"] = "0"
    inter.loc[inter.index[pick(3)], "timestamp"] = "yesterday"
    inter.loc[inter.index[pick(4)], "user_id"] = None
    inter.loc[inter.index[pick(6)], "item_id"] = "ORPHAN_ITEM"
    inter.loc[inter.index[pick(6)], "user_id"] = "ORPHAN_USER"
    inter = pd.concat([inter, inter.iloc[pick(8)]], ignore_index=True)
    users = pd.concat([users, users.iloc[[0, 1, 2]]], ignore_index=True)
    items.loc[items.index[:3], "item_id"] = None
    return users, items, inter

def run(cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    raw = path_for(cfg, "raw")
    out = raw / "faulty"
    out.mkdir(exist_ok=True)
    tables = [pd.read_csv(raw / f"{n}.csv", dtype=str) for n in ("users", "items", "interactions")]
    for name, df in zip(("users", "items", "interactions"), inject_faults(*tables, seed=cfg["project"]["seed"])):
        df.to_csv(out / f"{name}.csv", index=False)
    return str(out)

if __name__ == "__main__":
    print(run())
