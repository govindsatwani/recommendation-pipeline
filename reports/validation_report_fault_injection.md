# Data Validation Report - Fault-Injection Demo

Dataset: **Amazon Reviews 2018 - Software (5-core)**

Rules applied: missing values, data types, rating in [1, 5], timestamp in [1996-01-01, now], duplicate rows / keys, referential integrity (interaction -> user, item).

## Summary

| Table | Rows in | Valid | Rejected | Reject % |
|---|---:|---:|---:|---:|
| users | 1,829 | 1,826 | 3 | 0.16% |
| items | 26,790 | 21,636 | 5,154 | 19.24% |
| interactions | 12,813 | 11,941 | 872 | 6.81% |

## Rejected rows by rule

**users**

| Rule | Rows |
|---|---:|
| `duplicate_row` | 3 |

**items**

| Rule | Rows |
|---|---:|
| `duplicate_row` | 5,151 |
| `item_id:missing_value` | 3 |

**interactions**

| Rule | Rows |
|---|---:|
| `duplicate_row` | 823 |
| `rating:rating_in_range` | 10 |
| `timestamp:timestamp_in_range` | 10 |
| `referential_integrity:item_id_not_in_items` | 6 |
| `referential_integrity:user_id_not_in_users` | 6 |
| `rating:missing_value` | 5 |
| `rating:dtype_numeric` | 5 |
| `user_id:missing_value` | 4 |
| `timestamp:dtype_integer` | 3 |

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

Error datasets: `data/errors/faulty_demo`
