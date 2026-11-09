# Recommendation Pipeline for Software Products

**Name:** Govind Satwani  
**Roll No.:** 2025AG05056

This repository contains an end-to-end pipeline for a product recommendation system trained on the **Amazon Reviews 2018 - Software (5-core)** dataset (J. McAuley lab, UCSD).   
The pipeline ingests raw reviews and product metadata, validates them, prepares and transforms them, builds a feature layer, trains and evaluates two recommenders.  
The entire pipeline is orchestrated end to end with Prefect.

![architecture](docs/architecture.png)

## Stages and where to find them

| Stage | Code | Main outputs |
|---|---|---|
| 1. Ingestion | `src/ingestion/ingest.py` | `data/raw/users.csv`, `data/raw/items.csv`, `data/raw/interactions.csv`, `data/raw/ingestion_manifest.json` |
| 2. Validation | `src/validation/validate.py`, `fault_injection.py` | `data/processed/validated/*.parquet`, `data/errors/*_errors.csv`, `reports/validation_report*.md` |
| 3. Preparation | `src/preparation/prepare.py` | `*_clean.parquet`, `train.parquet`, `test.parquet`, `split_info.json` |
| 4. Transformation | `src/transformation/transform.py` | `user_item_matrix.npz`, `user_activity_summary.parquet`, `item_popularity.parquet` |
| 5. Feature management | `src/features/` | `data/features/*.parquet`, `reports/feature_catalog.md` |
| 6. Training | `src/training/` | `models/*.joblib`, MLflow runs |
| 7. Evaluation | `src/evaluation/` | `reports/model_comparison.md`, `reports/model_comparison.csv`, `reports/model_comparison.json` |
| 8. Orchestration | `dags/recsys_flow.py` | task run states in the console / Prefect UI, `logs/pipeline.log` |
| Serving | `src/inference/`, `predict.py` | `reports/sample_recommendations.md` |

Models: **popularity baseline** and **SVD matrix factorisation** (personalised). 
1. The final report can be found in [reports/final_report.md](reports/final_report.md).
2. The feature documentation can be found in [reports/feature_catalog.md](reports/feature_catalog.md).

## Setup

1. **Python 3.14** on Linux has been used for the code.
2. The pinned versions in `requirements.txt` have been installed and used.
3. The first run requires the dataset to be downloaded (size ~20MB) - hence the first run takes 3-4 minutes to download the dataset.
4. Subsequent runs use the cached version of the dataset, and start processing the data instantly.

### Command to set up the environment and install all dependencies and libraries
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
## Starting from a clean state (optional)

This is not a required step, because every stage overwrites its outputs.  
This can be used to reproduce the run from scratch, including the dataset download. Howver, the MLflow and Prefect servers need to be stopped first.  
The below commands can be run to clean up the files from the previous runs.  

```bash
rm -f data/raw/*.gz data/raw/items.zip data/raw/*.csv data/raw/ingestion_manifest.json
rm -rf data/raw/faulty data/errors data/processed/* data/features/*.parquet data/features/feature_metadata.json
rm -rf models/*.joblib models/runs mlflow.db mlartifacts logs
find reports -type f ! -name final_report.md -delete
```
Then run `.venv/bin/python dags/recsys_flow.py` as mentioned below.

## Single command to run the whole pipeline

```bash
export MLFLOW_DISABLE_AGENT_HINT=1
.venv/bin/python dags/recsys_flow.py
```

The flow runs: ingest -> validate -> prepare -> transform -> features -> train popularity and SVD models -> evaluate -> demo recommendations.  
Runs are deterministic (seed 42): two consecutive runs give identical metrics.  

The code is also capable of scheduled execution. Prefect only fires schedules against a real server, so `prefect server start` command needs to be run first and the flow needs to be served against it.  
The below command can be used to schedule a run everyday at 02:00 AM UTC:

```bash
.venv/bin/prefect server start  # execute on current open terminal
PREFECT_API_URL=http://127.0.0.1:4200/api .venv/bin/python dags/recsys_flow.py --serve "0 2 * * *"  # open a new terminal and then execute
```

UIs:

```bash
.venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000   # go to http://127.0.0.1:5000 on the browser -> experiment "software-recommender"
.venv/bin/prefect server start       # go to http://127.0.0.1:4200 on the browser
```

Individual stages of the pipeline can also be run on their own:

```bash
.venv/bin/python -m src.ingestion.ingest
.venv/bin/python -m src.validation.validate
.venv/bin/python -m src.preparation.prepare
.venv/bin/python -m src.transformation.transform
.venv/bin/python -m src.features.feature_store
.venv/bin/python -m src.training.train
.venv/bin/python -m src.evaluation.evaluate
.venv/bin/python -m src.inference.demo
```

## Get recommendations for a user

```bash
# existing user -> uses the reviews seen in training)
.venv/bin/python predict.py --user_id A2THQCHKR89E16 --k 10 --show-history

# brand-new user: pass their reviews as item_id:rating -> fold-in, no retraining
.venv/bin/python predict.py --history "B00004TYCR:5,B00003JAU7:4" --k 5

# unknown user without reviews -> popularity fallback
.venv/bin/python predict.py --user_id SOMEONE_NEW
```

The model can be chosen using the flag `--model {popularity,svd}` (default model is `svd`).

## Project layout

```text
recommendation-pipeline/
├── configs/
│   └── config.yaml # All parameters used throughout the process
├── dags/
│   └── recsys_flow.py # Complete Prefect flow
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── errors/
├── docs/
│   ├── screenshots/ # Screenshots for all the tools and steps
    └── architecture.png
├── models/
    └── runs/
├── reports/
    ├── feature_catalog.md
    ├── final_report.md
    ├── model_comparison.csv
    ├── model_comparison.json
    ├── model_comparison.md
    ├── sample_recommendations.md
    ├── validate_report_fault_injcetion.json
    ├── validate_report_fault_injcetion.md
    ├── validate_report.json
    └── validate_report.md
├── src/
    ├── evaluation/
    ├── features/
    ├── inference/
    ├── ingestion/
    ├── preparation/
    ├── training/
    ├── transformation/
    ├── validation/
    ├── __init__.py
    └── common.py # Ingestion, validation, preparation, transformation, features, training, evaluation, inference
├── .gitignore
├── predict.py # CLI recommender
├── README.md
└── requirements.txt

```

## Notes and limitations

* Evaluation is offline (time-based 80/20 split, K=10, relevant = rating >= 4). No live A/B test is being done.
* The dataset is small and sparse (density ~0.8%); about half of the test interactions involve items that were never seen in training and cannot be recommended by collaborative models.
* The user table only holds an id and name; whereas the user attributes are derived from interaction history.
* `data/raw/items.csv`, `data/processed/`, `data/features/*.parquet`, `models/*.joblib`, `mlflow.db`/`mlartifacts/` and `logs/` are git-ignored and rebuilt by the pipeline.
* The downloaded `.gz` files, `data/raw/items.zip` (zipped because the CSV is 41 MB), `models/runs/*.json`, `data/errors/`, `reports/` and `docs/screenshots/` are committed as evidence of my own run. A rerun will overwrite them.
* The small raw CSVs, the error datasets (`data/errors/`) and all generated reports (`reports/`) are committed for visibility into the operation of the pipeline.
* Screenshots for all the tools and executed steps of the pipeline can be found [here](docs/screenshots).
