# Feature Catalog

All features are computed from **training data only**

## User features

| Feature | Type | Description | Formula / logic | Source |
|---|---|---|---|---|
| `user_avg_rating` | float | Average rating the user has given. | `mean(rating) over the user's training interactions` | train.parquet |
| `user_rating_std` | float | Spread of the user's ratings (0 for a single rating). | `std(rating)` | train.parquet |
| `user_activity_count` | int | Number of interactions (reviews) by the user. | `count(interactions)` | train.parquet |
| `user_verified_ratio` | float | Share of the user's reviews that are verified purchases. | `mean(verified)` | train.parquet |
| `user_days_since_last_interaction` | float | Recency of the user's last interaction. | `(as_of - max(timestamp)) / 86400` | train.parquet |
| `user_preferred_category` | string | Category the user interacted with most (ties: higher total rating). | `argmax count(category)` | train + items_clean |
| `user_preferred_brand` | string | Brand the user interacted with most (ties: higher total rating). | `argmax count(brand)` | train + items_clean |
| `user_avg_price` | float | Average price of the items the user interacted with. | `mean(item.price)` | train + items_clean |

## Item features

| Feature | Type | Description | Formula / logic | Source |
|---|---|---|---|---|
| `item_popularity` | int | Number of training interactions the item received. | `count(interactions)` | train.parquet |
| `item_unique_users` | int | Number of distinct users who interacted with the item. | `nunique(user_id)` | train.parquet |
| `item_avg_rating` | float | Average rating of the item (global mean if never rated). | `mean(rating)` | train.parquet |
| `item_popularity_rank` | int | Rank of the item by popularity (1 = most popular). | `rank(item_popularity, descending)` | train.parquet |
| `item_days_since_last_interaction` | float | Recency of the last interaction with the item. | `(as_of - max(timestamp)) / 86400` | train.parquet |
| `item_price` | float | Cleaned price (missing prices imputed with category median). | `parse(price) -> median impute` | items_clean |
| `item_category` | string | Top-level product category. | `first level of the category path after 'Software'` | items_clean |
| `item_subcategory` | string | Second-level product category (may be empty). | `second level of the category path` | items_clean |
| `item_brand` | string | Brand of the item ('Unknown' if missing). | `cleaned brand text` | items_clean |

## Interaction features

| Feature | Type | Description | Formula / logic | Source |
|---|---|---|---|---|
| `interaction_category_match` | int | 1 if the item's category equals the user's preferred category. | `item_category == user_preferred_category` | user + item features |
| `interaction_brand_match` | int | 1 if the item's brand equals the user's preferred brand. | `item_brand == user_preferred_brand` | user + item features |
| `interaction_price_ratio` | float | Item price relative to what the user usually pays. | `item_price / user_avg_price` | user + item features |
| `interaction_rating_minus_user_avg` | float | How much the rating deviates from the user's average (training only). | `rating - user_avg_rating` | train.parquet |
| `interaction_days_before_as_of` | float | Age of the interaction at the reference date. | `(as_of - timestamp) / 86400` | train.parquet |
