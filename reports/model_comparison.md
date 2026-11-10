# Model Evaluation Results

* Dataset: Amazon Reviews 2018 - Software (5-core)
* Split: time-based, train < 2015-10-04T09:36:00+00:00 <= test
* K = 10; an item is *relevant* if the test rating >= 4
* Evaluated users: 308 (users with training history and >=1 relevant future item in the training catalog that was not previously seen by that user)
* Rating evaluation pairs: 703 (known users and items; includes repeat interactions)

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0052 | 0.0203 | 0.0115 | 0.0208 | 0.8536 | 1.3723 | 1.1166 |
| svd | 0.0156 | 0.1040 | 0.0686 | 0.3781 | 0.8047 | 1.1972 | 0.9643 |