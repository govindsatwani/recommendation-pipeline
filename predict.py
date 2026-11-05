# Command-line recommender
import argparse

import pandas as pd

from src.inference.recommend import DEFAULT_MODEL, MODELS, RecommendationService


def parse_history(text: str) -> dict[str, float]:
    out = {}
    for part in text.split(","):
        item, rating = part.split(":")
        out[item.strip()] = float(rating)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Top-K item recommendations for a user")
    p.add_argument("--user_id")
    p.add_argument("--history", help="new-user reviews as 'item_id:rating,item_id:rating'")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    p.add_argument("--show-history", action="store_true", help="print the user's past reviews first")
    a = p.parse_args()
    if not a.user_id and not a.history:
        p.error("provide --user_id and/or --history")

    pd.set_option("display.width", 200, "display.max_colwidth", 70, "display.max_columns", 20)
    svc = RecommendationService(model=a.model)
    if a.show_history and a.user_id:
        print(f"\nPast reviews of {a.user_id}:\n{svc.user_history(a.user_id).to_string(index=False)}")
    recs = svc.recommend(a.user_id, a.k, parse_history(a.history) if a.history else None)
    print(f"\nTop-{a.k} recommendations - source: {recs.attrs['source']}\n")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
