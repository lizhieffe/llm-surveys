"""Sibling of fetch_dolci_think_rl_source_samples_duckdb.py for
allenai/Dolci-Instruct-RL (the RLVR mixture behind Olmo 3 Instruct 7B's RL
stage).

Like Think-RL-7B, this card gives two breakdowns -- but unlike Think-RL-7B,
they aren't two views of the same complete data: the "Dataset Source Counts
(Grouped Mixes)" table (8 mixes) sums exactly to the card's stated 169,964
total, while the "Original Dataset Contribution" table (4 rows) sums to
only 85,966 -- it's a *partial* breakdown of just one of the 8 mixes
("General RLVR Mix", 48,398) plus a duplicate of the IF Multi-Constraint
count. A full local GROUP BY confirms which schema column backs which:

- `dataset_source` (8 unique values, no nulls) == the complete Grouped-Mixes
  split. This is the dataset's real, fully-covering categorization axis.
- `original_dataset` is **NULL for 83,998 of 169,964 rows** (the five
  Math/Code mixes never populate it) and only breaks down the "General
  RLVR Mix" bucket into its real 3 components (Multi-Subject RLVR + Tulu 3
  Rewritten + WildChat English, summing exactly to 48,398).

So the categorization here uses `dataset_source` for 7 of the 8 mixes
directly, but swaps in `original_dataset` for "General RLVR Mix" alone,
splitting it into its 3 finer sub-categories where the data actually
supports it -- 10 categories total, each verified to match live counts
exactly (37,568 + 18,971 + 18,757 + 10,670 + 20,000 + 20,000 + 14,000 +
14,000 + 8,998 + 7,000 = 169,964). The lesson generalizes: a dataset's
"finest available" granularity can vary *per row*, not just per dataset --
check for NULL coverage before assuming a more granular column is usable
everywhere.
"""

import json
import os
import urllib.request
from pathlib import Path

import duckdb

DATASET = "allenai/Dolci-Instruct-RL"
HF_TOKEN = os.environ.get("HF_TOKEN")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "dolci-samples"
NUM_SHARDS = 3
CACHE_DIR = Path(
    os.environ.get(
        "DOLCI_INSTRUCT_RL_PARQUET_CACHE",
        "/tmp/claude-1000/-home-lizhi-developments-cs336-assignment5-alignment/"
        "33a4e6bc-871f-4837-afe1-6e9a29ff0efe/scratchpad/dolci-instruct-rl-parquet",
    )
)
SEED = 42
TRUNCATE_CHARS = 4000
TRUNCATE_LIST_ITEMS = 5

# Human-readable columns only -- input_ids_prompt (a raw tokenized int
# array) is excluded, same rationale as Dolci-Think-RL-7B.
SELECT_COLUMNS = [
    "id", "custom_id", "prompt", "ground_truth", "dataset", "solution",
    "difficulty", "difficulty_explanation", "setting_key", "setting_name",
    "data_source", "ability", "reward_model", "topic", "characters",
    "original_dataset", "dataset_source", "outputs", "total_rollouts",
    "total_correct_rollouts", "passrate", "key", "constraint_type",
    '"constraint"', "conversation_hash", "model", "predicted_label",
]
RESULT_KEYS = [c.strip('"') for c in SELECT_COLUMNS]

# Each entry: slug, display title, which column to filter on
# ("dataset_source" for 7 of the 8 card mixes; "original_dataset" for the
# 3 categories split out of "General RLVR Mix", where dataset_source alone
# is too coarse -- see module docstring), the value to match, a repo_id to
# link to, license, citation, and the live-verified count.
DOLCI_INSTRUCT_RL_SOURCES = [
    {
        "slug": "if-multi-constraint-instruct-rl",
        "title": "IF Multi-Constraint",
        "filter_column": "dataset_source",
        "filter_value": "allenai/IF_multi_constraints_upto5_filtered_dpo_0625_filter-keyword-filtered-topic-char-topic-filtered",
        "repo_id": "allenai/IF_multi_constraints_upto5",
        "license": "ODC-BY-1.0",
        "row_count": 37568,
        "citation": None,
        "note": "Derived from IFBench-Train / IFEval-style prompts; up to 5 constraints, normalized and filtered for safety/clarity.",
    },
    {
        "slug": "multi-subject-rlvr-instruct-rl",
        "title": "Multi-Subject RLVR",
        "filter_column": "original_dataset",
        "filter_value": "hamishivi/virtuoussy_multi_subject_rlvr_filtered",
        "repo_id": "virtuoussy/Multi-subject-RLVR",
        "license": "Apache 2.0",
        "row_count": 18971,
        "citation": "arXiv:2503.23829v1",
        "note": "One of 3 sources combined into the card's \"General RLVR Mix\"; exam-style reasoning questions.",
    },
    {
        "slug": "tulu-3-rewritten-instruct-rl",
        "title": "Tulu 3 Rewritten",
        "filter_column": "original_dataset",
        "filter_value": "hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2_filtered",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "row_count": 18757,
        "citation": "Lambert et al., 2024 (arXiv:2411.15124)",
        "note": "One of 3 sources combined into the card's \"General RLVR Mix\"; Tulu 3 SFT prompts rewritten and F1-filtered, no separate upstream repo.",
    },
    {
        "slug": "wildchat-english-instruct-rl",
        "title": "WildChat English General",
        "filter_column": "original_dataset",
        "filter_value": "hamishivi/new-wildchat-english-general_filtered",
        "repo_id": "allenai/WildChat-1M",
        "license": "ODC-BY-1.0",
        "row_count": 10670,
        "citation": "arXiv:2405.01470",
        "note": "One of 3 sources combined into the card's \"General RLVR Mix\"; filtered for non-math/non-code, character caps applied.",
    },
    {
        "slug": "acecoder-instruct-rl",
        "title": "AceCoder RLVR",
        "filter_column": "dataset_source",
        "filter_value": "hamishivi/rlvr_acecoder_filtered_filtered",
        "repo_id": "TIGER-Lab/AceCode-87K",
        "license": "MIT",
        "row_count": 20000,
        "citation": "arXiv:2502.01718",
        "note": "Test-case-based RL prompts, filtered via solution execution.",
    },
    {
        "slug": "omega-math-instruct-rl",
        "title": "OMEGA (Math)",
        "filter_column": "dataset_source",
        "filter_value": "hamishivi/omega-combined-no-boxed_filtered",
        "repo_id": "allenai/omega-explorative",
        "license": "MIT",
        "row_count": 20000,
        "citation": "arXiv:2506.18880",
        "note": "Combined across OMEGA's explorative, compositional, and transformative subsets.",
    },
    {
        "slug": "orz-math-instruct-rl",
        "title": "ORZ Math (Open-Reasoner-Zero)",
        "filter_column": "dataset_source",
        "filter_value": "hamishivi/rlvr_orz_math_57k_collected_filtered",
        "repo_id": "Open-Reasoner-Zero/orz_math_57k_collection",
        "license": "MIT",
        "row_count": 14000,
        "citation": "arXiv:2503.24290",
        "note": None,
    },
    {
        "slug": "polaris-math-instruct-rl",
        "title": "Polaris Math",
        "filter_column": "dataset_source",
        "filter_value": "hamishivi/polaris_53k",
        "repo_id": "POLARIS-Project/Polaris-Dataset-53K",
        "license": "Apache 2.0",
        "row_count": 14000,
        "citation": None,
        "note": None,
    },
    {
        "slug": "mathsub-30k-instruct-rl",
        "title": "MathSub-30K (KlearReasoner Math)",
        "filter_column": "dataset_source",
        "filter_value": "hamishivi/MathSub-30K_filtered",
        "repo_id": "Kwai-Klear/KlearReasoner-MathSub-30K",
        "license": "Apache 2.0",
        "row_count": 8998,
        "citation": "arXiv:2508.07629",
        "note": None,
    },
    {
        "slug": "dapo-math-instruct-rl",
        "title": "DAPO-Math",
        "filter_column": "dataset_source",
        "filter_value": "hamishivi/DAPO-Math-17k-Processed_filtered",
        "repo_id": "BytedTsinghua-SIA/DAPO-Math-17k",
        "license": "Apache 2.0",
        "row_count": 7000,
        "citation": "arXiv:2503.14476",
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
    return f"dolci-instruct-rl-source__{slug}.json"


def main() -> None:
    download_shards()
    glob = str(CACHE_DIR / "*.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "dolci_instruct_rl_source_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    select_clause = ", ".join(SELECT_COLUMNS)

    for i, src in enumerate(DOLCI_INSTRUCT_RL_SOURCES, 1):
        slug = src["slug"]
        out_path = SAMPLES_DIR / safe_filename(slug)
        if out_path.exists() and status.get(slug) == "ok":
            print(f"[{i}/{len(DOLCI_INSTRUCT_RL_SOURCES)}] skip (cached) {slug}")
            continue

        column = src["filter_column"]
        value = src["filter_value"].replace("'", "''")
        print(f"[{i}/{len(DOLCI_INSTRUCT_RL_SOURCES)}] querying {slug!r} ({column}) ...", end=" ", flush=True)

        try:
            (num_rows,) = con.execute(
                f"SELECT count(*) FROM read_parquet('{glob}') WHERE {column} = '{value}'"
            ).fetchone()
            if num_rows == 0:
                status[slug] = "no_rows"
                print("no_rows")
                continue

            con.execute("SELECT setseed(?)", [SEED / 2147483647])
            rows_raw = con.execute(f"""
                SELECT {select_clause}
                FROM read_parquet('{glob}')
                WHERE {column} = '{value}'
                ORDER BY random()
                LIMIT {min(16, num_rows)}
            """).fetchall()

            rows = [truncate(dict(zip(RESULT_KEYS, r))) for r in rows_raw]
            result = {
                "status": "ok",
                "config": "default",
                "split": "train",
                "features": [
                    {"feature_idx": idx, "name": name, "type": {"_type": "Value"}}
                    for idx, name in enumerate(RESULT_KEYS)
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
    print(f"\nDone. {ok}/{len(DOLCI_INSTRUCT_RL_SOURCES)} sources sampled successfully.")


if __name__ == "__main__":
    main()
