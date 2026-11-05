# Data Validation Report

Dataset: **Amazon Reviews 2018 - Software (5-core)**

Rules applied: missing values, data types, rating in [1, 5], timestamp in [1996-01-01, now], duplicate rows / keys, referential integrity (interaction -> user, item).

## Summary

| Table | Rows in | Valid | Rejected | Reject % |
|---|---:|---:|---:|---:|
| users | 1,826 | 1,826 | 0 | 0.00% |
| items | 26,790 | 21,639 | 5,151 | 19.23% |
| interactions | 12,805 | 11,984 | 821 | 6.41% |

## Rejected rows by rule

**users**

No rows rejected.

**items**

| Rule | Rows |
|---|---:|
| `duplicate_row` | 5,151 |

**interactions**

| Rule | Rows |
|---|---:|
| `duplicate_row` | 821 |

## Warnings (rows kept, handled in preparation)

| Check | Count |
|---|---:|
| interactions.repeated_user_item_pairs | 163 |
| items.missing_title | 6 |
| items.missing_brand | 659 |
| items.missing_categories | 2,063 |
| items.missing_price | 20,182 |
| items.unparseable_price | 1,290 |
| items.missing_description | 3,597 |
| users.missing_user_name | 1 |

Error datasets: `data/errors`
