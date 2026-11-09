# Command-line recommender
import argparse

import pandas as pd

from src.inference.recommend import DEFAULT_MODEL, MODELS, RecommendationService

def parse_history(text: str) -> dict[str, float]:
    out = {}

    for part in text.split(","):
        item, sep, rating = part.partition(":")
        item = item.strip()

        if not sep or not item or not rating.strip():
            raise ValueError("Each review must use item_id:rating")

        if item in out:
            raise ValueError(f"Duplicate item in history: {item}")

        try:
            out[item] = float(rating)
        except ValueError as exc:
            raise ValueError(f"Invalid rating for item {item}: {rating!r}") from exc
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
    try:
        history = (parse_history(a.history) if a.history is not None else None)
        recs = svc.recommend(a.user_id, a.k, history)
    except ValueError as exc:
        p.error(str(exc))
    print(f"\nTop-{a.k} recommendations - source: {recs.attrs['source']}\n")
    print(recs.to_string(index=False))

if __name__ == "__main__":
    main()
