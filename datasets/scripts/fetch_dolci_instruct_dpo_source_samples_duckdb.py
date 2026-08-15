"""Sibling of fetch_dolci_think_dpo_source_samples_duckdb.py for
allenai/Dolci-Instruct-DPO (the preference-tuning mixture behind Olmo 3
Instruct 7B's DPO stage).

Unlike every other Dolci dataset covered so far, this one has **no
dataset_source / source_dataset column at all** -- rows aren't tagged with
which upstream prompt collection they came from. Instead, the card
describes the mixture as a blend of *preference-construction methods*
applied to (largely) the same underlying prompt pool, and the row schema
carries that split directly via `preference_type`:

    "The Dolci Instruct DPO mixture ... contains 260,000 preference pairs
    in total, including: 125,000 pairs created with the preference
    heuristic described in Delta Learning (Geng et al. 2025); 125,000
    pairs created with a delta-aware Ultrafeedback-esque GPT-judge
    pipeline ...; 10,000 multiturn preference pairs (5,000 synthetic
    context, 5,000 self talk)."

A full local `GROUP BY preference_type` confirms this almost exactly (the
card's figures are rounded): 124,980 llm_judged + 124,942 delta_learning +
5,000 multiturn_self_talk + 5,000 multiturn_synthetic_context = 259,922,
matching the live row count exactly. So `preference_type` -- not a
source-dataset column -- is this dataset's native categorization axis.

Row shape also differs from the other DPO dataset (Dolci-Think-DPO-7B):
`chosen`/`rejected` are lists of much richer WildChat-style turn objects
(hashed_ip, country, language, request headers, etc. -- the same kind of
metadata WildChat-1M already publishes; this dataset carries no source
column at all, so there's nothing extra to redact here beyond what's
already public in the upstream data), not simple {content, role} messages,
and there's a `prompt_id` instead of `id`. No `dataset_source` field to
select at all.
"""

import json
import os
import urllib.request
from pathlib import Path

import duckdb

DATASET = "allenai/Dolci-Instruct-DPO"
HF_TOKEN = os.environ.get("HF_TOKEN")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "dolci-samples"
NUM_SHARDS = 4
CACHE_DIR = Path(
    os.environ.get(
        "DOLCI_INSTRUCT_DPO_PARQUET_CACHE",
        "/tmp/claude-1000/-home-lizhi-developments-cs336-assignment5-alignment/"
        "33a4e6bc-871f-4837-afe1-6e9a29ff0efe/scratchpad/dolci-instruct-dpo-parquet",
    )
)
SEED = 42
TRUNCATE_CHARS = 4000
TRUNCATE_LIST_ITEMS = 5

# Each entry: slug, display title, the `preference_type` value to match,
# license (ODC-BY-1.0 per the card, same as the rest of Dolci), citation
# where the card names a specific method, and the live-verified row count
# (there's no separate card-stated count to cross-check per category
# beyond the card's rounded totals, already confirmed in the module
# docstring).
DOLCI_INSTRUCT_DPO_SOURCES = [
    {
        "slug": "llm-judged-instruct-dpo",
        "title": "LLM-Judged (delta-aware, UltraFeedback-style)",
        "preference_type_value": "llm_judged",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "row_count": 124980,
        "citation": None,
        "note": "Delta-aware Ultrafeedback-esque GPT-judge pipeline, designed to maximize chosen/rejected contrast.",
    },
    {
        "slug": "delta-learning-instruct-dpo",
        "title": "Delta Learning",
        "preference_type_value": "delta_learning",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "row_count": 124942,
        "citation": "Geng et al., 2025 (arXiv:2507.06187)",
        "note": None,
    },
    {
        "slug": "multiturn-synthetic-context-instruct-dpo",
        "title": "Multiturn (synthetic context)",
        "preference_type_value": "multiturn_synthetic_context",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "row_count": 5000,
        "citation": None,
        "note": None,
    },
    {
        "slug": "multiturn-self-talk-instruct-dpo",
        "title": "Multiturn (self talk)",
        "preference_type_value": "multiturn_self_talk",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "row_count": 5000,
        "citation": None,
        "note": None,
    },
]


def shard_url(i: int) -> str:
    return f"https://huggingface.co/api/datasets/{DATASET}/parquet/default/train/{i}.parquet"


def download_shards() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "llm-surveys"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

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


def truncate(value):
    if isinstance(value, str) and len(value) > TRUNCATE_CHARS:
        return value[:TRUNCATE_CHARS] + f"... [truncated, {len(value):,} chars total]"
    if isinstance(value, list):
        kept = [truncate(v) for v in value[:TRUNCATE_LIST_ITEMS]]
        if len(value) > TRUNCATE_LIST_ITEMS:
            kept.append(f"... [{len(value) - TRUNCATE_LIST_ITEMS} more items omitted, {len(value)} total]")
        return kept
    if isinstance(value, dict):
        return {k: truncate(v) for k, v in value.items()}
    return value


def safe_filename(slug: str) -> str:
    return f"dolci-instruct-dpo-source__{slug}.json"


def main() -> None:
    download_shards()
    glob = str(CACHE_DIR / "*.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "dolci_instruct_dpo_source_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    for i, src in enumerate(DOLCI_INSTRUCT_DPO_SOURCES, 1):
        slug = src["slug"]
        out_path = SAMPLES_DIR / safe_filename(slug)
        if out_path.exists() and status.get(slug) == "ok":
            print(f"[{i}/{len(DOLCI_INSTRUCT_DPO_SOURCES)}] skip (cached) {slug}")
            continue

        value = src["preference_type_value"].replace("'", "''")
        print(f"[{i}/{len(DOLCI_INSTRUCT_DPO_SOURCES)}] querying {slug!r} ...", end=" ", flush=True)

        try:
            (num_rows,) = con.execute(
                f"SELECT count(*) FROM read_parquet('{glob}') WHERE preference_type = '{value}'"
            ).fetchone()
            if num_rows == 0:
                status[slug] = "no_rows"
                print("no_rows")
                continue

            con.execute("SELECT setseed(?)", [SEED / 2147483647])
            rows_raw = con.execute(f"""
                SELECT prompt_id, chosen, rejected, chosen_model, rejected_model, preference_type
                FROM read_parquet('{glob}')
                WHERE preference_type = '{value}'
                ORDER BY random()
                LIMIT {min(16, num_rows)}
            """).fetchall()

            rows = [
                truncate({
                    "prompt_id": r[0],
                    "chosen": r[1],
                    "rejected": r[2],
                    "chosen_model": r[3],
                    "rejected_model": r[4],
                    "preference_type": r[5],
                })
                for r in rows_raw
            ]
            result = {
                "status": "ok",
                "config": "default",
                "split": "train",
                "features": [
                    {"feature_idx": 0, "name": "prompt_id", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 1, "name": "chosen", "type": {"_type": "List"}},
                    {"feature_idx": 2, "name": "rejected", "type": {"_type": "List"}},
                    {"feature_idx": 3, "name": "chosen_model", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 4, "name": "rejected_model", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 5, "name": "preference_type", "type": {"dtype": "string", "_type": "Value"}},
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
    print(f"\nDone. {ok}/{len(DOLCI_INSTRUCT_DPO_SOURCES)} sources sampled successfully.")


if __name__ == "__main__":
    main()
