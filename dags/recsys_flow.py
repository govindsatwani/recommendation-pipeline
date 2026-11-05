# Complete orchestration of the full recommendation pipeline using Prefect
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefect import flow, get_run_logger, task

from src.common import load_config, set_seed
from src.evaluation import evaluate as evaluate_stage
from src.features import feature_store
from src.inference import demo
from src.ingestion import ingest as ingest_stage
from src.preparation import prepare as prepare_stage
from src.training import train as train_stage
from src.transformation import transform as transform_stage
from src.validation import fault_injection, validate as validate_stage

RETRY = dict(retries=1, retry_delay_seconds=5)

@task(name="1-ingest", **RETRY)
def ingest() -> dict:
    m = ingest_stage.run(load_config())
    return {k: v["rows"] for k, v in m["files"].items() if "rows" in v}

@task(name="2-validate")
def validate(upstream: dict) -> dict:
    rep = validate_stage.run(load_config(), source="raw")
    get_run_logger().info("Validation: %s", {t: v["rows_rejected"] for t, v in rep["tables"].items()})
    return {t: v["rows_rejected"] for t, v in rep["tables"].items()}

@task(name="2b-validation-fault-injection-demo")
def validation_fault_injection_demo(upstream: dict) -> dict:
    cfg = load_config()
    fault_injection.run(cfg)
    rep = validate_stage.run(cfg, source="faulty")
    return {t: v["rows_rejected"] for t, v in rep["tables"].items()}

@task(name="3-prepare")
def prepare(upstream: dict) -> dict:
    return prepare_stage.run(load_config())

@task(name="4-transform")
def transform(upstream: dict) -> dict:
    return transform_stage.run(load_config())

@task(name="5-features")
def features(upstream: dict) -> dict:
    return feature_store.run(load_config())

@task(name="6a-train-popularity")
def train_popularity(upstream: dict) -> str:
    return train_stage.train_model("popularity", load_config())

@task(name="6b-train-svd")
def train_svd(upstream: dict) -> str:
    return train_stage.train_model("svd", load_config())

@task(name="7-evaluate")
def evaluate(run_ids: list) -> dict:
    return evaluate_stage.run(load_config())

@task(name="8-demo-recommendations")
def demo_recommendations(results: dict) -> str:
    return demo.run(load_config())

@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline() -> dict:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    raw = ingest()
    validated = validate(raw)
    fault_demo = validation_fault_injection_demo(raw) # injecting intentional faults to test validation process
    prepared = prepare(validated)
    transformed = transform(prepared)
    feats = features(transformed)
    runs = [train_popularity(feats), train_svd(feats)]
    results = evaluate(runs)
    demo_recommendations(results)
    print("Validation rejections on injected faults:", fault_demo)
    print("Pipeline finished. Results:", results)
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", metavar="CRON", help="serve the flow on a cron schedule instead of running once")
    args = ap.parse_args()
    if args.serve:
        recommendation_pipeline.serve(name="recsys-scheduled", cron=args.serve)
    else:
        recommendation_pipeline()
