# Ranking and rating metrics
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

def precision_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / k

def recall_at_k(recs: list, relevant: set, k: int) -> float:
    return len(set(recs[:k]) & relevant) / len(relevant) if relevant else 0.0

def ndcg_at_k(recs: list, relevant: set, k: int) -> float: # Binary-relevance NDCG@k
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recs[:k]) if item in relevant)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0

def coverage(all_recs: list[list], catalog_size: int) -> float:
    return len({i for recs in all_recs for i in recs}) / catalog_size

def intra_list_diversity(rec_rows: list[list[int]], attr_matrix: sp.csr_matrix) -> float:
    vals = []
    for rows in rec_rows:
        if len(rows) < 2:
            continue
        sub = attr_matrix[rows]
        sims = (sub @ sub.T).toarray()
        n = len(rows)
        vals.append(1.0 - (sims.sum() - np.trace(sims)) / (n * (n - 1))) # Average pairwise cosine dissimilarity of the items inside each list
    return float(np.mean(vals)) if vals else 0.0

def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))
