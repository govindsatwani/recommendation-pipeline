# Model training with MLflow experiment tracking
from __future__ import annotations

import hashlib
import json
import time

import joblib
import mlflow

from src.common import ROOT, get_logger, load_config, path_for, set_seed
from src.training.models import (
    PopularityRecommender,
    SVDRecommender,
    load_training_data,
)

MODEL_NAMES = ("popularity", "svd")

def setup_mlflow(cfg: dict) -> None:
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"].replace("sqlite:///", f"sqlite:///{ROOT}/"))
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(name)

def _file_hash(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]

def train_model(name: str, cfg: dict | None = None) -> str:
    # Train one model, log it to MLflow and then returning the MLflow run id
    cfg = cfg or load_config()
    log = get_logger("training", cfg)
    set_seed(cfg["project"]["seed"])
    proc, models_dir = path_for(cfg, "processed"), path_for(cfg, "models")
    info = json.loads((proc / "split_info.json").read_text())
    data = load_training_data(cfg)
    setup_mlflow(cfg)

    model = {
        "popularity": lambda: PopularityRecommender(),
        "svd": lambda: SVDRecommender(cfg["models"]["svd"]["n_factors"], cfg["models"]["svd"]["rating_factors"]),
    }[name]()
    params = {
        "dataset": cfg["dataset"]["name"],
        "split_cutoff": info["cutoff_date"],
        "train_rows": info["train_rows"],
        "n_users": len(data.user_ids),
        "n_items": len(data.item_ids),
        "seed": cfg["project"]["seed"],
        "train_data_hash": _file_hash(proc / "train.parquet"),
        "feature_as_of": data.store.as_of,
    }
    with mlflow.start_run(run_name=name) as run:
        t0 = time.perf_counter()
        model.fit(data, cfg)
        fit_s = time.perf_counter() - t0
        mlflow.log_params({**params, **model.get_params()})
        mlflow.log_metric("fit_seconds", fit_s)
        path = models_dir / f"{name}.joblib"
        joblib.dump(model, path)
        mlflow.log_artifact(str(path), artifact_path="model")
        mlflow.log_artifact(str(proc / "split_info.json"))
        mlflow.log_artifact(str(path_for(cfg, "reports") / "feature_catalog.md"))
        run_id = run.info.run_id
    runs_dir = models_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / f"{name}.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    log.info("Trained %-10s in %.2fs -> %s (run %s)", name, fit_s, path.name, run_id[:8])
    return run_id

def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return {n: train_model(n, cfg) for n in MODEL_NAMES}

if __name__ == "__main__":
    run()
