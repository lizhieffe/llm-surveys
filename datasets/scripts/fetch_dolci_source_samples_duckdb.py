"""Alternative to fetch_dolci_source_samples.py: instead of hitting HF's
datasets-server /filter endpoint per-row (374 small requests, flaky while
its DuckDB index warms up across backend replicas), download the
Dolci-Instruct-SFT parquet shards once and query them locally with our own
DuckDB. Produces the exact same output format
(dolci-samples/dolci-source__<slug>.json), so build_dolci_source_manifest.py
doesn't need to change.

Rationale / tradeoffs are discussed in the chat that produced this script:
the datasets-server path transfers very little data (~1-3KB/row) but is slow
due to backend-side retries; streaming parquet directly over httpfs re-pays
a full-column scan cost per query and hits HF's file-CDN rate limit. A single
local download of the ~3GB dataset, then 22 local queries, avoids both
problems at the cost of one upfront 3GB transfer and ~3GB disk.
"""

import json
import os
import random
import urllib.request
from pathlib import Path

import duckdb

from fetch_dolci_source_samples import (
    DATA_DIR,
    HF_TOKEN,
    SAMPLES_DIR,
    SOURCES,
    safe_filename,
    truncate,
)

NUM_SHARDS = 15
DATASET = "allenai/Dolci-Instruct-SFT"
CACHE_DIR = Path(
    os.environ.get(
        "DOLCI_PARQUET_CACHE",
        "/tmp/claude-1000/-home-lizhi-developments-cs336-assignment5-alignment/"
        "33a4e6bc-871f-4837-afe1-6e9a29ff0efe/scratchpad/dolci-parquet",
    )
)
SEED = 42


def shard_url(i: int) -> str:
    return f"https://huggingface.co/api/datasets/{DATASET}/parquet/default/train/{i}.parquet"


def download_shards() -> list[str]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "llm-surveys"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    paths = []
    for i in range(NUM_SHARDS):
        dest = CACHE_DIR / f"{i}.parquet"
        url = shard_url(i)
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            expected_size = int(resp.headers.get("Content-Length", -1))

        if dest.exists() and dest.stat().st_size == expected_size:
            print(f"shard {i}: cached ({expected_size:,} bytes)")
        else:
            print(f"shard {i}: downloading ({expected_size:,} bytes) ...", end=" ", flush=True)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            print("done")
        paths.append(str(dest))
    return paths


def main() -> None:
    shard_paths = download_shards()
    glob = str(CACHE_DIR / "*.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA threads=1")  # deterministic single-threaded scan for reproducible sampling

    status_path = DATA_DIR / "dolci_source_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    for i, src in enumerate(SOURCES, 1):
        slug = src["slug"]
        out_path = SAMPLES_DIR / safe_filename(slug)
        if out_path.exists() and status.get(slug) == "ok":
            print(f"[{i}/{len(SOURCES)}] skip (cached) {slug}")
            continue

        print(f"[{i}/{len(SOURCES)}] querying {slug!r} (source_dataset={src['source_dataset']!r}) ...",
              end=" ", flush=True)
        source_value = src["source_dataset"].replace("'", "''")

        try:
            (num_rows,) = con.execute(
                f"SELECT count(*) FROM read_parquet('{glob}') WHERE source_dataset = '{source_value}'"
            ).fetchone()
            if num_rows == 0:
                status[slug] = "no_rows"
                print("no_rows")
                continue

            con.execute("SELECT setseed(?)", [SEED / 2147483647])
            rows_raw = con.execute(f"""
                SELECT id, messages, source_dataset, domain
                FROM read_parquet('{glob}')
                WHERE source_dataset = '{source_value}'
                ORDER BY random()
                LIMIT {min(16, num_rows)}
            """).fetchall()

            rows = [
                truncate({"id": r[0], "messages": r[1], "source_dataset": r[2], "domain": r[3]})
                for r in rows_raw
            ]
            result = {
                "status": "ok",
                "config": "default",
                "split": "train",
                "features": [
                    {"feature_idx": 0, "name": "id", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 1, "name": "messages", "type": {"_type": "List"}},
                    {"feature_idx": 2, "name": "source_dataset", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 3, "name": "domain", "type": {"dtype": "string", "_type": "Value"}},
                ],
                "rows": rows,
                "num_rows_total": num_rows,
            }
            out_path.write_text(json.dumps(result, indent=2, default=str))
            status[slug] = "ok"
            print(f"ok ({num_rows:,} total rows, sampled {len(rows)})")
        except Exception as e:  # noqa: BLE001
            status[slug] = "error"
            print(f"error: {e}")

        status_path.write_text(json.dumps(status, indent=2))

    ok = sum(1 for v in status.values() if v == "ok")
    print(f"\nDone. {ok}/{len(SOURCES)} sources sampled successfully.")


if __name__ == "__main__":
    main()
