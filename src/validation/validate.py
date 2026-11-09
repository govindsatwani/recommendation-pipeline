# Data validation
from __future__ import annotations

import json
import re

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from src.common import ROOT, get_logger, load_config, path_for

PRICE_RE = re.compile(r"^\$?\s*\d{1,6}(?:,\d{3})*(?:\.\d{1,2})?$")

def _numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").notna()

def _integer_like(s: pd.Series) -> pd.Series:
    n = pd.to_numeric(s, errors="coerce")
    return n.notna() & (n == n.round())

def _valid_vote(s: pd.Series) -> pd.Series:
    valid_format = s.str.fullmatch(
        r"(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)",
        na=False,
    )

    result = pd.Series(False, index=s.index)

    normalized = (
        s[valid_format]
        .str.replace(",", "", regex=False)
        .str.lstrip("0")
    )
    normalized = normalized.mask(normalized == "", "0")

    result.loc[valid_format] = (
        (normalized.str.len() < 19)
        | (
            (normalized.str.len() == 19)
            & (normalized <= "9223372036854775807")
        )
    )

    return result

def build_schemas(cfg: dict) -> dict[str, pa.DataFrameSchema]:
    v = cfg["validation"]
    ts_min = pd.Timestamp(v["timestamp_min"], tz="UTC").timestamp()
    ts_max = pd.Timestamp.now(tz="UTC").timestamp() if v["timestamp_max"] == "now" else pd.Timestamp(
        v["timestamp_max"], tz="UTC"
    ).timestamp()

    def rating_range(s: pd.Series) -> pd.Series:
        n = pd.to_numeric(s, errors="coerce")
        return n.isna() | ((n >= v["rating_min"]) & (n <= v["rating_max"]))

    def ts_range(s: pd.Series) -> pd.Series:
        n = pd.to_numeric(s, errors="coerce")
        return n.isna() | ((n >= ts_min) & (n <= ts_max))

    interactions = pa.DataFrameSchema(
        {
            "user_id": pa.Column(str, nullable=False),
            "item_id": pa.Column(str, nullable=False),
            "rating": pa.Column(
                str,
                nullable=False,
                checks=[
                    pa.Check(_numeric, name="dtype_numeric"),
                    pa.Check(rating_range, name="rating_in_range"),
                ],
            ),
            "timestamp": pa.Column(
                str,
                nullable=False,
                checks=[
                    pa.Check(_integer_like, name="dtype_integer"),
                    pa.Check(ts_range, name="timestamp_in_range"),
                ],
            ),
            "verified": pa.Column(
                str, nullable=False, checks=pa.Check.isin(["True", "False", "true", "false"], name="dtype_boolean")
            ),
            "vote": pa.Column(
                str,
                nullable=True,
                checks=pa.Check(
                    _valid_vote,
                    name="nonnegative_integer",
                ),
            ),
            "summary": pa.Column(str, nullable=True),
        },
        strict=False,
    )
    users = pa.DataFrameSchema(
        {"user_id": pa.Column(str, nullable=False), "user_name": pa.Column(str, nullable=True)}, strict=False
    )
    items = pa.DataFrameSchema(
        {
            "item_id": pa.Column(str, nullable=False),
            "title": pa.Column(str, nullable=True),
            "brand": pa.Column(str, nullable=True),
            "main_category": pa.Column(str, nullable=True),
            "categories": pa.Column(str, nullable=True),
            "price": pa.Column(str, nullable=True),
            "description": pa.Column(str, nullable=True),
        },
        strict=False,
    )
    return {"interactions": interactions, "users": users, "items": items}

def _schema_errors(df: pd.DataFrame, schema: pa.DataFrameSchema) -> dict[int, list[str]]:
    reasons: dict[int, list[str]] = {}
    try:
        schema.validate(df, lazy=True)
    except SchemaErrors as exc:
        fc = exc.failure_cases
        for _, row in fc.iterrows():
            idx = row["index"]
            if pd.isna(idx):
                continue
            check = str(row["check"])
            if check.startswith("not_nullable"):
                check = "missing_value"
            reasons.setdefault(int(idx), []).append(f"{row['column']}:{check}")
    return reasons

def _flag(errors: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    new = errors.copy()
    for idx in mask[mask].index:
        new.at[idx] = (new.at[idx] + ";" + reason) if new.at[idx] else reason
    return new

def validate_tables(raw: dict[str, pd.DataFrame], cfg: dict) -> tuple[dict, dict, dict]:
    schemas = build_schemas(cfg)
    report: dict = {"tables": {}, "warnings": {}}
    flags: dict[str, pd.Series] = {}

    for name, df in raw.items():
        df = df.copy()
        raw[name] = df
        err = pd.Series("", index=df.index, dtype=object)
        for idx, rs in _schema_errors(df, schemas[name]).items():
            err.at[idx] = ";".join(sorted(set(rs)))
        # exact duplicates
        err = _flag(err, df.duplicated(keep="first"), "duplicate_row")
        flags[name] = err

    users, items, inter = raw["users"], raw["items"], raw["interactions"]

    # users/items duplicated by key but with different content
    for name, key in (("users", "user_id"), ("items", "item_id")):
        df = raw[name]
        ok = flags[name] == ""
        dup_key = df[key].duplicated(keep="first") & ok
        flags[name] = _flag(flags[name], dup_key, f"duplicate_{key}")

    # referential integrity
    valid_users = set(users.loc[flags["users"] == "", "user_id"])
    valid_items = set(items.loc[flags["items"] == "", "item_id"])
    ok = flags["interactions"] == ""
    flags["interactions"] = _flag(
        flags["interactions"], ok & ~inter["user_id"].isin(valid_users), "referential_integrity:user_id_not_in_users"
    )
    ok = flags["interactions"] == ""
    flags["interactions"] = _flag(
        flags["interactions"], ok & ~inter["item_id"].isin(valid_items), "referential_integrity:item_id_not_in_items"
    )

    pair_dups = inter.loc[flags["interactions"] == ""].duplicated(["user_id", "item_id"], keep=False)
    warn = {
        "interactions.repeated_user_item_pairs": int(pair_dups.sum()),
        "items.missing_title": int(items["title"].isna().sum()),
        "items.missing_brand": int(items["brand"].isna().sum()),
        "items.missing_categories": int(items["categories"].isna().sum()),
        "items.missing_price": int(items["price"].isna().sum()),
        "items.unparseable_price": int(
            (items["price"].notna() & ~items["price"].fillna("").str.strip().str.match(PRICE_RE)).sum()
        ),
        "items.missing_description": int(items["description"].isna().sum()),
        "users.missing_user_name": int(users["user_name"].isna().sum()),
    }
    report["warnings"] = warn

    clean, errors = {}, {}
    for name, df in raw.items():
        bad = flags[name] != ""
        clean[name] = df.loc[~bad].copy()
        e = df.loc[bad].copy()
        e["error_reason"] = flags[name][bad]
        errors[name] = e
        by_rule: dict[str, int] = {}
        for r in flags[name][bad]:
            for part in r.split(";"):
                by_rule[part] = by_rule.get(part, 0) + 1
        report["tables"][name] = {
            "rows_in": int(len(df)),
            "rows_valid": int((~bad).sum()),
            "rows_rejected": int(bad.sum()),
            "rejected_by_rule": dict(sorted(by_rule.items(), key=lambda kv: -kv[1])),
        }
    return clean, errors, report

def _cast_clean(clean: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    i = clean["interactions"].copy()
    i["rating"] = pd.to_numeric(i["rating"]).astype("float64")
    i["timestamp"] = pd.to_numeric(i["timestamp"]).astype("int64")
    i["verified"] = i["verified"].str.lower().eq("true")
    i["vote"] = (i["vote"].str.replace(",", "", regex=False).astype("Int64"))
    clean = {**clean, "interactions": i.reset_index(drop=True)}
    clean["users"] = clean["users"].reset_index(drop=True)
    clean["items"] = clean["items"].reset_index(drop=True)
    return clean

# automatically creating reports
def render_markdown(report: dict, title: str, cfg: dict) -> str:
    v = cfg["validation"]
    lines = [f"# {title}", ""]
    lines += [
        f"Dataset: **{cfg['dataset']['name']}**",
        "",
        "Rules applied: missing values, data types, rating in "
        f"[{v['rating_min']}, {v['rating_max']}], timestamp in [{v['timestamp_min']}, now], "
        "duplicate rows / keys, referential integrity (interaction -> user, item).",
        "",
        "## Summary",
        "",
        "| Table | Rows in | Valid | Rejected | Reject % |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, t in report["tables"].items():
        pct = 100 * t["rows_rejected"] / max(t["rows_in"], 1)
        lines.append(f"| {name} | {t['rows_in']:,} | {t['rows_valid']:,} | {t['rows_rejected']:,} | {pct:.2f}% |")
    lines += ["", "## Rejected rows by rule", ""]
    for name, t in report["tables"].items():
        lines.append(f"**{name}**")
        lines.append("")
        if not t["rejected_by_rule"]:
            lines += ["No rows rejected.", ""]
            continue
        lines += ["| Rule | Rows |", "|---|---:|"]
        lines += [f"| `{r}` | {c:,} |" for r, c in t["rejected_by_rule"].items()]
        lines.append("")
    lines += [
        "## Warnings (rows kept, handled in preparation)",
        "",
        "| Check | Count |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {c:,} |" for k, c in report["warnings"].items()]
    lines += ["", f"Error datasets: `{report.get('error_dir', 'data/errors')}`", ""]
    return "\n".join(lines)

def run(cfg: dict | None = None, source: str = "raw") -> dict:
    cfg = cfg or load_config()
    log = get_logger("validation", cfg)
    raw_dir = path_for(cfg, "raw") if source == "raw" else path_for(cfg, "raw") / "faulty"
    err_dir = path_for(cfg, "errors") if source == "raw" else path_for(cfg, "errors") / "faulty_demo"
    err_dir.mkdir(parents=True, exist_ok=True)
    out_dir = path_for(cfg, "validated")
    rep_dir = path_for(cfg, "reports")

    raw = {n: pd.read_csv(raw_dir / f"{n}.csv", dtype=str) for n in ("users", "items", "interactions")}
    for n, df in raw.items():
        log.info("Loaded %-12s rows=%d", n, len(df))

    clean, errors, report = validate_tables(raw, cfg)
    report["source"] = str(raw_dir.relative_to(ROOT))
    report["error_dir"] = str(err_dir.relative_to(ROOT))

    for n, e in errors.items():
        e.to_csv(err_dir / f"{n}_errors.csv", index=False)
        t = report["tables"][n]
        log.info(
            "%-12s valid=%d rejected=%d %s", n, t["rows_valid"], t["rows_rejected"], json.dumps(t["rejected_by_rule"])
        )
    for k, c in report["warnings"].items():
        log.warning("warning %s = %d", k, c)

    if source == "raw":
        clean = _cast_clean(clean)
        for n, df in clean.items():
            df.to_parquet(out_dir / f"{n}.parquet", index=False)
        stem, title = "validation_report", "Data Validation Report"
    else:
        stem, title = "validation_report_fault_injection", "Data Validation Report - Fault-Injection Demo"
    with open(rep_dir / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    (rep_dir / f"{stem}.md").write_text(render_markdown(report, title, cfg), encoding="utf-8")
    log.info("Validation report written to reports/%s.md", stem)
    return report

if __name__ == "__main__":
    import sys

    run(source=sys.argv[1] if len(sys.argv) > 1 else "raw")
