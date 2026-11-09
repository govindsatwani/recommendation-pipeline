from __future__ import annotations

import json
import time

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.common import ROOT, get_logger, load_config, path_for
from src.evaluation import metrics as M
from src.training.models import TrainData, item_attribute_matrix, load_training_data, top_k

MODEL_ORDER = ["popularity", "svd"]

def build_eval_set(test: pd.DataFrame, data: TrainData, threshold: float):
    known = test[
        test["user_id"].isin(data.user_idx)
        & test["item_id"].isin(data.item_idx)
    ]

    rows = known["user_id"].map(data.user_idx).to_numpy(dtype=int)
    cols = known["item_id"].map(data.item_idx).to_numpy(dtype=int)
    seen = np.asarray(data.R[rows, cols]).ravel() != 0

    rankable = known.loc[~seen]
    relevant = (
        rankable[rankable["rating"] >= threshold]
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )
    return known, relevant

def evaluate_model(model, data: TrainData, attrs, test: pd.DataFrame, cfg: dict) -> dict:
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    pairs, relevant = build_eval_set(test, data, thr)
    if not relevant:
        raise ValueError("No unseen relevant test items available for ranking evaluation")

    users = sorted(relevant)
    rows = np.array([data.user_idx[u] for u in users])
    R_rows = data.R[rows]

    t0 = time.perf_counter()
    idx, _ = top_k(model.score(R_rows), R_rows, k)
    rec_items = [[data.item_ids[j] for j in r] for r in idx]
    latency_ms = (time.perf_counter() - t0) * 1000 / max(len(users), 1)

    res = {
        f"precision@{k}": np.mean([M.precision_at_k(r, relevant[u], k) for r, u in zip(rec_items, users)]),
        f"recall@{k}": np.mean([M.recall_at_k(r, relevant[u], k) for r, u in zip(rec_items, users)]),
        f"ndcg@{k}": np.mean([M.ndcg_at_k(r, relevant[u], k) for r, u in zip(rec_items, users)]),
        "coverage": M.coverage(rec_items, len(data.item_ids)),
        "diversity": M.intra_list_diversity([list(r) for r in idx], attrs),
    }
    preds = model.predict_rating(pairs["user_id"].to_numpy(), pairs["item_id"].to_numpy())
    res["rmse"] = M.rmse(pairs["rating"], preds) if preds is not None else None
    res["mae"] = M.mae(pairs["rating"], preds) if preds is not None else None
    res["eval_users"] = len(users)
    res["eval_rating_pairs"] = len(pairs)
    res["latency_ms_per_user"] = latency_ms
    return {k_: (float(v) if v is not None else None) for k_, v in res.items()}

def _fmt(v, nd=4):
    return "n/a" if v is None or pd.isna(v) else f"{v:.{nd}f}"

# automatically creating reports
def results_markdown(df: pd.DataFrame, cfg: dict, ctx: dict) -> str:
    k = cfg["evaluation"]["k"]
    cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}", "coverage", "diversity", "rmse", "mae"]
    lines = [
        "# Model Evaluation Results",
        "",
        f"* Dataset: {cfg['dataset']['name']}",
                f"* Split: time-based, train < {ctx['cutoff_time']} <= test",
        f"* K = {k}; an item is *relevant* if the test rating >= {cfg['evaluation']['relevance_threshold']}",
        f"* Evaluated users: {ctx['eval_users']} (users with training history and >=1 relevant future item "
        f"in the training catalog that was not previously seen by that user)",
        f"* Rating evaluation pairs: {ctx['eval_rating_pairs']} "
        f"(known users and items; includes repeat interactions)",
        "",
        "| Model | " + " | ".join(c.replace("precision", "Precision").replace("recall", "Recall").replace("ndcg", "NDCG")
                                  .replace("coverage", "Coverage").replace("diversity", "Diversity")
                                  .replace("rmse", "RMSE").replace("mae", "MAE") for c in cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for name, row in df.iterrows():
        lines.append(f"| {name} | " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)

def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("evaluation", cfg)
    proc, models_dir, rep = path_for(cfg, "processed"), path_for(cfg, "models"), path_for(cfg, "reports")
    data = load_training_data(cfg)
    test = pd.read_parquet(proc / "test.parquet")
    info = json.loads((proc / "split_info.json").read_text())
    attrs = item_attribute_matrix(data)
    run_ids = {n: json.loads((models_dir / "runs" / f"{n}.json").read_text())["run_id"] for n in MODEL_ORDER}

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT}/"))
    results = {}
    for name in MODEL_ORDER:
        model = joblib.load(models_dir / f"{name}.joblib")
        res = evaluate_model(model, data, attrs, test, cfg)
        results[name] = res
        with mlflow.start_run(run_id=run_ids[name]):
            mlflow.log_metrics({f"test_{k.replace('@', '_at_')}": v for k, v in res.items() if v is not None})
        log.info("%-10s %s", name, {k: round(v, 4) for k, v in res.items() if v is not None})

    df = pd.DataFrame(results).T
    df.index.name = "model"
    df.to_csv(rep / "model_comparison.csv")
    ctx = {
        "cutoff_time": pd.to_datetime(
            info["cutoff_timestamp"], unit="s", utc=True
        ).isoformat(),
        "eval_users": int(df["eval_users"].iloc[0]),
        "eval_rating_pairs": int(df["eval_rating_pairs"].iloc[0]),
    }
    (rep / "model_comparison.md").write_text(results_markdown(df, cfg, ctx), encoding="utf-8")
    (rep / "model_comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name in MODEL_ORDER:
        with mlflow.start_run(run_id=run_ids[name]):
            mlflow.log_artifact(str(rep / "model_comparison.md"))
    log.info("Results written to reports/model_comparison.md")
    return results

if __name__ == "__main__":
    run()
