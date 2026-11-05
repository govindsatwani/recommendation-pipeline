# Model Evaluation Results

* Dataset: Amazon Reviews 2018 - Software (5-core)
* Split: time-based, train < 2015-09-30 <= test
* K = 10; an item is *relevant* if the test rating >= 4
* Evaluated users: 311 (users with training history and >=1 relevant future item in the training catalog)

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0058 | 0.0233 | 0.0138 | 0.0208 | 0.8536 | 1.3752 | 1.1192 |
| svd | 0.0161 | 0.1046 | 0.0703 | 0.3781 | 0.8037 | 1.1953 | 0.9637 |