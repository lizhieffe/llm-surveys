"""Sibling of fetch_dolci_think_source_samples_duckdb.py for
allenai/Dolci-Think-RL-7B (the RLVR mixture behind Olmo 3 Think 7B's RL
stage).

Unlike the other Dolci datasets, this card gives *two* breakdowns at once:
a coarse 4-way "Grouped Mixes" split (Math/IF/Code/General RLVR Mixture)
and a fine 13-way "Original Dataset Contribution" split (IF Multi-Constraint,
OMEGA Math, AceCoder, ...). Both sum exactly to the card's stated 102,014
total. A full local GROUP BY confirms which schema column maps to which:

- `dataset_source` (4 unique values) == the coarse Grouped-Mixes split.
- `original_dataset` (15 unique values) == the fine split, at *finer*
  granularity than the card's 13 named categories -- the card's single
  "Llama-Nemotron Post-Training Dataset" (2,006) is actually 3 separate
  original_dataset values (difficulty-6/7/8 RLVR subsets) that sum to
  exactly that count, the same "several raw values combine into one
  category" pattern seen in Dolci-Think-SFT-7B's OpenThoughts 3.

Per the site's categorization rule (finest meaningful granularity), this
script uses `original_dataset` -- the fine 13-category split -- not
`dataset_source`'s coarse 4-mix grouping.

Row schema is much wider than the SFT/DPO datasets (RLVR training fields:
ground_truth, outputs, rollout counts/passrate, tokenized input_ids/labels/
attention_mask for the actual RL training run, plus per-task fields like
constraint/constraint_type for IF rows or model/conversation_hash for
WildChat-derived rows, mostly null outside their own task type). Sample
rows here select only the human-readable fields -- the tokenized
input_ids/attention_mask/labels arrays are excluded entirely (meaningless
without the training tokenizer, and would be thousands of raw ints per
row bloating sample files for no benefit).
"""

import json
import os
import urllib.request
from pathlib import Path

import duckdb

DATASET = "allenai/Dolci-Think-RL-7B"
HF_TOKEN = os.environ.get("HF_TOKEN")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "dolci-samples"
NUM_SHARDS = 9
CACHE_DIR = Path(
    os.environ.get(
        "DOLCI_THINK_RL_PARQUET_CACHE",
        "/tmp/claude-1000/-home-lizhi-developments-cs336-assignment5-alignment/"
        "33a4e6bc-871f-4837-afe1-6e9a29ff0efe/scratchpad/dolci-think-rl-parquet",
    )
)
SEED = 42
TRUNCATE_CHARS = 4000
TRUNCATE_LIST_ITEMS = 5

# Non-tokenizer, human-readable columns only -- see module docstring for
# why input_ids/attention_mask/labels/input_ids_prompt are excluded.
SELECT_COLUMNS = [
    "id", "custom_id", "prompt", "ground_truth", "dataset", "original_dataset",
    "outputs", "total_rollouts", "total_correct_rollouts", "passrate",
    "dataset_source", "key", "constraint_type", '"constraint"',
    "conversation_hash", "model", "predicted_label",
]
RESULT_KEYS = [c.strip('"') for c in SELECT_COLUMNS]

# Each entry: slug, display title, the raw `original_dataset` value(s) to
# match (a list -- the card's "Llama-Nemotron Post-Training Dataset" is 3
# raw values that sum to its stated count), a repo_id to link to (the
# upstream source repo where a clear one was found via HF search, else
# this dataset itself), license, citation (arXiv links given directly on
# the Dolci-Think-RL-7B card per source), and the card-stated count
# (verified against the live GROUP BY -- see module docstring).
DOLCI_THINK_RL_SOURCES = [
    {
        "slug": "if-multi-constraint-rl",
        "title": "IF Multi-Constraint",
        "original_dataset_values": ["hamishivi/IF_multi_constraints_upto5_filtered"],
        "repo_id": "allenai/IF_multi_constraints_upto5",
        "license": "ODC-BY-1.0",
        "card_count": 29813,
        "citation": None,
        "note": "Up to 5 constraints per prompt; derived from IFBench-Train / IFEval-style tasks, filtered for clarity and non-toxicity.",
    },
    {
        "slug": "omega-math-rl",
        "title": "OMEGA Math",
        "original_dataset_values": ["hamishivi/omega-combined-no-boxed_filtered"],
        "repo_id": "allenai/omega-explorative",
        "license": "MIT",
        "card_count": 15000,
        "citation": "arXiv:2506.18880",
        "note": "Combined across OMEGA's explorative, compositional, and transformative subsets.",
    },
    {
        "slug": "acecoder-rl",
        "title": "AceCoder",
        "original_dataset_values": ["hamishivi/rlvr_acecoder_filtered_filtered"],
        "repo_id": "TIGER-Lab/AceCode-87K",
        "license": "MIT",
        "card_count": 10107,
        "citation": "arXiv:2502.01718",
        "note": "Filtered via test-case execution.",
    },
    {
        "slug": "tulu-3-rewritten-rl",
        "title": "Tulu 3 Rewritten",
        "original_dataset_values": [
            "hamishivi/tulu_3_rewritten_400k_string_f1_only_v2_nocode_all_filtered_qwen2_5_openthoughts2_filtered",
        ],
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 7109,
        "citation": "Lambert et al., 2024 (arXiv:2411.15124)",
        "note": "Tulu 3 SFT prompts rewritten and F1-filtered; no separate upstream repo.",
    },
    {
        "slug": "multi-subject-rlvr",
        "title": "Multi-Subject RLVR",
        "original_dataset_values": ["hamishivi/virtuoussy_multi_subject_rlvr_filtered"],
        "repo_id": "virtuoussy/Multi-subject-RLVR",
        "license": "Apache 2.0",
        "card_count": 7106,
        "citation": "arXiv:2503.23829v1",
        "note": None,
    },
    {
        "slug": "acereason-math-rl",
        "title": "AceReason-Math",
        "original_dataset_values": ["hamishivi/AceReason-Math_filtered"],
        "repo_id": "nvidia/AceReason-Math",
        "license": "CC-BY-4.0",
        "card_count": 6598,
        "citation": "arXiv:2505.16400",
        "note": None,
    },
    {
        "slug": "wildchat-english-rl",
        "title": "WildChat English",
        "original_dataset_values": ["hamishivi/new-wildchat-english-general_filtered"],
        "repo_id": "allenai/WildChat-1M",
        "license": "ODC-BY-1.0",
        "card_count": 6421,
        "citation": "arXiv:2405.01470",
        "note": "Filtered for reasoning suitability.",
    },
    {
        "slug": "klearreasoner-code-rl",
        "title": "KlearReasoner Code",
        "original_dataset_values": ["hamishivi/klear-code-rlvr_filtered"],
        "repo_id": "Kwai-Klear/KlearReasoner-CodeSub-15K",
        "license": "Apache 2.0",
        "card_count": 6272,
        "citation": "arXiv:2508.07629",
        "note": "Filtered via test-case execution.",
    },
    {
        "slug": "synthetic-2-rl",
        "title": "SYNTHETIC-2 / PrimeIntellect",
        "original_dataset_values": ["hamishivi/synthetic2-rlvr-code-compressed_filtered"],
        "repo_id": "PrimeIntellect/SYNTHETIC-2-RL",
        "license": None,
        "card_count": 3000,
        "citation": None,
        "note": "Filtered via test-case execution (see PrimeIntellect's SYNTHETIC-2 blog post).",
    },
    {
        "slug": "mathsub-30k-rl",
        "title": "MathSub-30K (KlearReasoner Math)",
        "original_dataset_values": ["hamishivi/MathSub-30K_filtered"],
        "repo_id": "Kwai-Klear/KlearReasoner-MathSub-30K",
        "license": "Apache 2.0",
        "card_count": 2999,
        "citation": "arXiv:2508.07629",
        "note": None,
    },
    {
        "slug": "orz-math-rl",
        "title": "ORZ Math",
        "original_dataset_values": ["hamishivi/rlvr_orz_math_57k_collected_filtered"],
        "repo_id": "Open-Reasoner-Zero/orz_math_57k_collection",
        "license": "MIT",
        "card_count": 2999,
        "citation": "arXiv:2503.24290",
        "note": None,
    },
    {
        "slug": "dapo-math-rl",
        "title": "DAPO-Math",
        "original_dataset_values": ["hamishivi/DAPO-Math-17k-Processed_filtered"],
        "repo_id": "BytedTsinghua-SIA/DAPO-Math-17k",
        "license": "Apache 2.0",
        "card_count": 2584,
        "citation": "arXiv:2503.14476",
        "note": None,
    },
    {
        "slug": "llama-nemotron-post-training-rl",
        "title": "Llama-Nemotron Post-Training Dataset",
        "original_dataset_values": [
            "hamishivi/llama-nemotron-rlvr-difficulty-6_filtered",
            "hamishivi/llama-nemotron-rlvr-difficulty-7_filtered",
            "hamishivi/llama-nemotron-rlvr-difficulty-8_filtered",
        ],
        "repo_id": "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "license": "CC-BY-4.0",
        "card_count": 2006,
        "citation": "arXiv:2505.00949",
        "note": "Difficulty-tiered subsets (6/7/8) combined; filtered by difficulty tier.",
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
    return f"dolci-think-rl-source__{slug}.json"


def main() -> None:
    download_shards()
    glob = str(CACHE_DIR / "*.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "dolci_think_rl_source_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    select_clause = ", ".join(SELECT_COLUMNS)

    for i, src in enumerate(DOLCI_THINK_RL_SOURCES, 1):
        slug = src["slug"]
        out_path = SAMPLES_DIR / safe_filename(slug)
        if out_path.exists() and status.get(slug) == "ok":
            print(f"[{i}/{len(DOLCI_THINK_RL_SOURCES)}] skip (cached) {slug}")
            continue

        values = src["original_dataset_values"]
        in_list = ", ".join("'" + v.replace("'", "''") + "'" for v in values)
        print(f"[{i}/{len(DOLCI_THINK_RL_SOURCES)}] querying {slug!r} ({len(values)} value(s)) ...",
              end=" ", flush=True)

        try:
            (num_rows,) = con.execute(
                f"SELECT count(*) FROM read_parquet('{glob}') WHERE original_dataset IN ({in_list})"
            ).fetchone()
            if num_rows == 0:
                status[slug] = "no_rows"
                print("no_rows")
                continue

            con.execute("SELECT setseed(?)", [SEED / 2147483647])
            rows_raw = con.execute(f"""
                SELECT {select_clause}
                FROM read_parquet('{glob}')
                WHERE original_dataset IN ({in_list})
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
    print(f"\nDone. {ok}/{len(DOLCI_THINK_RL_SOURCES)} sources sampled successfully.")


if __name__ == "__main__":
    main()
