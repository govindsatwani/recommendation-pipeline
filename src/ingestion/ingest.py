# Data ingestion
from __future__ import annotations

import gzip
import hashlib
import json
import ssl
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from src.common import get_logger, load_config, path_for

class _RelaxedStrictTLS(HTTPAdapter):

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kwargs["ssl_context"] = ctx
        super().init_poolmanager(*args, **kwargs)

def _session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", _RelaxedStrictTLS())
    return s

def _download(url: str, dest: Path, log) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Using cached download %s", dest.name) # using cached downloads for runs other than the first one
        return
    log.info("Downloading %s", url) # downloading the dataset when the code is run for the first time
    with _session().get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dest)

def _read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def _join(value, sep: str = " | ") -> str | None:
    if isinstance(value, list):
        return sep.join(str(v) for v in value) if value else None
    return value

def build_raw_tables(reviews_path: Path, meta_path: Path) -> dict[str, pd.DataFrame]:
    review_rows, user_rows = [], {}
    for r in _read_jsonl_gz(reviews_path):
        review_rows.append(
            {
                "user_id": r.get("reviewerID"),
                "item_id": r.get("asin"),
                "rating": r.get("overall"),
                "timestamp": r.get("unixReviewTime"),
                "verified": r.get("verified"),
                "vote": r.get("vote"),
                "summary": r.get("summary"),
            }
        )
        user_rows.setdefault(r.get("reviewerID"), r.get("reviewerName"))
    interactions = pd.DataFrame(review_rows)
    users = pd.DataFrame({"user_id": list(user_rows.keys()), "user_name": list(user_rows.values())})

    item_rows = []
    for m in _read_jsonl_gz(meta_path):
        item_rows.append(
            {
                "item_id": m.get("asin"),
                "title": m.get("title"),
                "brand": m.get("brand"),
                "main_category": m.get("main_cat"),
                "categories": _join(m.get("category")),
                "price": m.get("price"),
                "description": _join(m.get("description"), " "),
            }
        )
    items = pd.DataFrame(item_rows)
    return {"users": users, "items": items, "interactions": interactions}

def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    log = get_logger("ingestion", cfg)
    raw_dir = path_for(cfg, "raw")
    ds = cfg["dataset"]

    reviews_gz, meta_gz = raw_dir / ds["reviews_file"], raw_dir / ds["meta_file"]
    _download(ds["reviews_url"], reviews_gz, log)
    _download(ds["meta_url"], meta_gz, log)

    tables = build_raw_tables(reviews_gz, meta_gz)
    manifest = {"dataset": ds["name"], "files": {}}
    for name, df in tables.items():
        out = raw_dir / f"{name}.csv"
        df.to_csv(out, index=False)
        manifest["files"][f"{name}.csv"] = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "sha256": _sha256(out),
        }
        log.info("Wrote %-16s rows=%d cols=%d", out.name, len(df), df.shape[1])

    for gz in (reviews_gz, meta_gz):
        manifest["files"][gz.name] = {"sha256": _sha256(gz), "bytes": gz.stat().st_size}
    with open(raw_dir / "ingestion_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Ingestion complete")
    return manifest

if __name__ == "__main__":
    run()
