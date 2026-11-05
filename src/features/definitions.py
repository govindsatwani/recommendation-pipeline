# Feature registry
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    entity: str
    dtype: str
    description: str
    formula: str
    source: str

FEATURES: list[FeatureDefinition] = [
    # user features
    FeatureDefinition("user_avg_rating", "user", "float", "Average rating the user has given.", "mean(rating) over the user's training interactions", "train.parquet"),
    FeatureDefinition("user_rating_std", "user", "float", "Spread of the user's ratings (0 for a single rating).", "std(rating)", "train.parquet"),
    FeatureDefinition("user_activity_count", "user", "int", "Number of interactions (reviews) by the user.", "count(interactions)", "train.parquet"),
    FeatureDefinition("user_verified_ratio", "user", "float", "Share of the user's reviews that are verified purchases.", "mean(verified)", "train.parquet"),
    FeatureDefinition("user_days_since_last_interaction", "user", "float", "Recency of the user's last interaction.", "(as_of - max(timestamp)) / 86400", "train.parquet"),
    FeatureDefinition("user_preferred_category", "user", "string", "Category the user interacted with most (ties: higher total rating).", "argmax count(category)", "train + items_clean"),
    FeatureDefinition("user_preferred_brand", "user", "string", "Brand the user interacted with most (ties: higher total rating).", "argmax count(brand)", "train + items_clean"),
    FeatureDefinition("user_avg_price", "user", "float", "Average price of the items the user interacted with.", "mean(item.price)", "train + items_clean"),
    # item features
    FeatureDefinition("item_popularity", "item", "int", "Number of training interactions the item received.", "count(interactions)", "train.parquet"),
    FeatureDefinition("item_unique_users", "item", "int", "Number of distinct users who interacted with the item.", "nunique(user_id)", "train.parquet"),
    FeatureDefinition("item_avg_rating", "item", "float", "Average rating of the item (global mean if never rated).", "mean(rating)", "train.parquet"),
    FeatureDefinition("item_popularity_rank", "item", "int", "Rank of the item by popularity (1 = most popular).", "rank(item_popularity, descending)", "train.parquet"),
    FeatureDefinition("item_days_since_last_interaction", "item", "float", "Recency of the last interaction with the item.", "(as_of - max(timestamp)) / 86400", "train.parquet"),
    FeatureDefinition("item_price", "item", "float", "Cleaned price (missing prices imputed with category median).", "parse(price) -> median impute", "items_clean"),
    FeatureDefinition("item_category", "item", "string", "Top-level product category.", "first level of the category path after 'Software'", "items_clean"),
    FeatureDefinition("item_subcategory", "item", "string", "Second-level product category (may be empty).", "second level of the category path", "items_clean"),
    FeatureDefinition("item_brand", "item", "string", "Brand of the item ('Unknown' if missing).", "cleaned brand text", "items_clean"),
    # interaction features
    FeatureDefinition("interaction_category_match", "interaction", "int", "1 if the item's category equals the user's preferred category.", "item_category == user_preferred_category", "user + item features"),
    FeatureDefinition("interaction_brand_match", "interaction", "int", "1 if the item's brand equals the user's preferred brand.", "item_brand == user_preferred_brand", "user + item features"),
    FeatureDefinition("interaction_price_ratio", "interaction", "float", "Item price relative to what the user usually pays.", "item_price / user_avg_price", "user + item features"),
    FeatureDefinition("interaction_rating_minus_user_avg", "interaction", "float", "How much the rating deviates from the user's average (training only).", "rating - user_avg_rating", "train.parquet"),
    FeatureDefinition("interaction_days_before_as_of", "interaction", "float", "Age of the interaction at the reference date.", "(as_of - timestamp) / 86400", "train.parquet"),
]

BY_ENTITY = {e: [f for f in FEATURES if f.entity == e] for e in ("user", "item", "interaction")}
