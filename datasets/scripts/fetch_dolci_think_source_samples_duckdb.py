"""Sibling of fetch_dolci_source_samples_duckdb.py for
allenai/Dolci-Think-SFT-7B (the SFT-stage reasoning-trace mixture behind
Olmo 3's Think models) -- same idea (one category per upstream sub-source,
16 sampled rows each, via local DuckDB against downloaded parquet shards
instead of the flaky datasets-server /filter endpoint), but two things
differ from Dolci-Instruct-SFT:

1. The row schema is different: `dataset_source` (not `source_dataset`), no
   `domain` column, and `messages` entries only have content/role (no
   function_calls/functions). See DOLCI_THINK_SOURCES below for the mapping.

2. `dataset_source` values are raw upstream-repo-style strings, not the
   clean human labels Dolci-Instruct-SFT uses, and they're *finer-grained*
   than the categories on the dataset card: e.g. the card's "OpenThoughts 3"
   (941,166 prompts) is actually 3 separate dataset_source values (math +
   science + code subsets) that sum to exactly that number, and "WildChat"
   (83,054) is 2 values that sum exactly too. Every category below was
   verified this way -- via a full local `GROUP BY dataset_source` across
   all 156 shards, matched against the card's stated per-category counts --
   not assumed from the card text alone (which, on its own, undercounts:
   the card doesn't state a `dataset_source`-taggable value for the "Olmo
   Identity Prompts" (58 rows) category, and the real GROUP BY confirms
   those 58 rows have no distinguishable tag in the released data -- adding
   58 to the other 13 categories' verified sum overshoots the real dataset
   total by exactly 58). That category is marked "no_rows" here rather than
   guessed at.

The dataset is much bigger than Dolci-Instruct-SFT (36GB across 156 parquet
shards vs. ~3GB across 15), so downloading it once and querying locally
(rather than streaming per-query over HTTP) matters even more here.
"""

import json
import os
import urllib.request
from pathlib import Path

import duckdb

DATASET = "allenai/Dolci-Think-SFT-7B"
HF_TOKEN = os.environ.get("HF_TOKEN")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "dolci-samples"
NUM_SHARDS = 156
CACHE_DIR = Path(
    os.environ.get(
        "DOLCI_THINK_PARQUET_CACHE",
        "/tmp/claude-1000/-home-lizhi-developments-cs336-assignment5-alignment/"
        "33a4e6bc-871f-4837-afe1-6e9a29ff0efe/scratchpad/dolci-think-parquet",
    )
)
SEED = 42
TRUNCATE_CHARS = 4000
TRUNCATE_LIST_ITEMS = 5

# Each entry: slug, display title, the raw `dataset_source` value(s) to
# match (a list -- some card categories are the sum of multiple raw
# values), license/citation as stated on the Dolci-Think-SFT-7B card (native
# Dolci sources are ODC-BY-1.0 per the card), the card's stated prompt
# count (verified against the live GROUP BY, see module docstring), and a
# repo_id to link to (the upstream source repo where one exists, else this
# dataset itself).
DOLCI_THINK_SOURCES = [
    {
        "slug": "openthoughts-3-think",
        "title": "OpenThoughts 3",
        "dataset_source_values": [
            "saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2",
            "saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2",
            "saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2",
        ],
        "repo_id": "open-thoughts/OpenThoughts3-1.2M",
        "license": "Apache 2.0",
        "card_count": 941166,
        "citation": None,
        "note": (
            "Extended to 32K context length and downsampled code prompts to 16x "
            "multiple; reasoning traces kept for the Think variant (math + science "
            "+ code subsets)."
        ),
    },
    {
        "slug": "synthetic-2-think",
        "title": "SYNTHETIC-2",
        "dataset_source_values": [
            "allenai/SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered",
        ],
        "repo_id": "PrimeIntellect/SYNTHETIC-2-SFT-verified",
        "license": "Apache 2.0",
        "card_count": 104569,
        "citation": None,
        "note": "Via the SFT-Verified split.",
    },
    {
        "slug": "nemotron-post-training-code-think",
        "title": "Nemotron Post-training (code)",
        "dataset_source_values": [
            "allenai/nemotron-post-training-dataset-subset-ngram-filtered-no-tool-calls",
        ],
        "repo_id": "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "license": "CC BY 4.0",
        "card_count": 113777,
        "citation": None,
        "note": "Code split only.",
    },
    {
        "slug": "dolci-think-persona-if",
        "title": "Dolci Think Persona IF",
        "dataset_source_values": [
            "allenai/persona-precise-if-r1-final-content-filtered-chinese-filtered",
        ],
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 223123,
        "citation": None,
        "note": (
            "New precise instruction-following prompts and traces created with "
            "Nvidia's Nemotron Post-training Personas; no separate upstream repo."
        ),
    },
    {
        "slug": "dolci-precise-if-think",
        "title": "Dolci Precise IF",
        "dataset_source_values": [
            "saumyamalik/if_qwq_reasoning_verified_filtered_decontam_v2",
        ],
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 135792,
        "citation": "Pyatkin et al., 2025 (arXiv:2507.02833)",
        "note": (
            "New multi-constraint instruction-following data building off "
            "“Generalizing Verifiable Instruction Following”; no separate "
            "upstream repo."
        ),
    },
    {
        "slug": "dolci-think-python",
        "title": "Dolci Think Python",
        "dataset_source_values": [
            "saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2",
        ],
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 466677,
        "citation": None,
        "note": "Subsampled from a larger mix; no separate upstream repo.",
    },
    {
        "slug": "wildchat-think",
        "title": "WildChat",
        "dataset_source_values": [
            "allenai/tulu_v3.9_wildchat_100k_english-r1-final-content-filtered",
            "allenai/wildchat-r1-p2-repetition-filter",
        ],
        "repo_id": "allenai/WildChat-1M",
        "license": "ODC-BY-1.0",
        "card_count": 83054,
        "citation": None,
        "note": "Original WildChat prompts, repurposed from Tulu 3 / OLMo 2 with new DeepSeek R1 traces.",
    },
    {
        "slug": "openassistant-guanaco-think",
        "title": "OpenAssistant Guanaco",
        "dataset_source_values": [
            "allenai/oasst1-r1-format-filtered-keyword-filtered-filter-datecutoff-chinese-filtered",
        ],
        "repo_id": "timdettmers/openassistant-guanaco",
        "license": "Apache 2.0",
        "card_count": 6800,
        "citation": None,
        "note": "Repurposed from Tulu 3 / OLMo 2 with new DeepSeek R1 traces.",
    },
    {
        "slug": "coconot-think",
        "title": "CoCoNot",
        "dataset_source_values": [
            "allenai/coconot-r1-format-domain-filtered-keyword-filtered-filter-datecutoff-chinese-filtered",
        ],
        "repo_id": "allenai/coconot",
        "license": "ODC-BY-1.0",
        "card_count": 10227,
        "citation": None,
        "note": "Repurposed from Tulu 3 / OLMo 2 with new DeepSeek R1 traces.",
    },
    {
        "slug": "wildguardmix-think",
        "title": "WildGuardMix",
        "dataset_source_values": [
            "allenai/wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered",
        ],
        "repo_id": "allenai/wildguardmix",
        "license": "Apache 2.0",
        "card_count": 38315,
        "citation": None,
        "note": "Repurposed from Tulu 3 / OLMo 2 with new DeepSeek R1 traces.",
    },
    {
        "slug": "wildjailbreak-think",
        "title": "WildJailbreak",
        "dataset_source_values": [
            "allenai/wildjailbreak-r1-v2-format-filtered-keyword-filtered-filter-datecutoff-final-content-filtered",
        ],
        "repo_id": "allenai/wildjailbreak",
        "license": "ODC-BY-1.0",
        "card_count": 41100,
        "citation": None,
        "note": "Repurposed from Tulu 3 / OLMo 2 with new DeepSeek R1 traces.",
    },
    {
        "slug": "aya-think",
        "title": "Aya",
        "dataset_source_values": [
            "allenai/aya-100k-r1-format-filtered-keyword-filtered-filter-datecutoff-ngram-filtered",
        ],
        "repo_id": "CohereForAI/aya_dataset",
        "license": "Apache 2.0",
        "card_count": 98597,
        "citation": None,
        "note": "Repurposed from Tulu 3 / OLMo 2 with new DeepSeek R1 traces.",
    },
    {
        "slug": "tablegpt-think",
        "title": "TableGPT",
        "dataset_source_values": [
            "allenai/tablegpt_r1-format-filtered-keyword-filtered-filter-datecutoff",
        ],
        "repo_id": "LipengCS/Table-GPT",
        "license": "MIT",
        "card_count": 4981,
        "citation": "Zha et al., 2023",
        "note": "Repurposed from Tulu 3 / OLMo 2 with new DeepSeek R1 traces.",
    },
    {
        "slug": "olmo-identity-prompts",
        "title": "Olmo Identity Prompts",
        "dataset_source_values": [],  # not separately taggable in the released data; see module docstring
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 58,
        "citation": None,
        "note": (
            "58 identity-Q&A samples (trained with 5x repetition, 290 total; a single "
            "repetition was uploaded). Not separately identifiable via `dataset_source` "
            "in the released parquet -- a full GROUP BY across all 2,268,178 rows finds "
            "no distinct value for it, so no sample rows are shown for this category."
        ),
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
    return f"dolci-think-source__{slug}.json"


def main() -> None:
    download_shards()
    glob = str(CACHE_DIR / "*.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "dolci_think_source_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    for i, src in enumerate(DOLCI_THINK_SOURCES, 1):
        slug = src["slug"]
        out_path = SAMPLES_DIR / safe_filename(slug)
        if out_path.exists() and status.get(slug) == "ok":
            print(f"[{i}/{len(DOLCI_THINK_SOURCES)}] skip (cached) {slug}")
            continue

        values = src["dataset_source_values"]
        if not values:
            status[slug] = "no_rows"
            print(f"[{i}/{len(DOLCI_THINK_SOURCES)}] {slug!r}: no taggable dataset_source, marking no_rows")
            status_path.write_text(json.dumps(status, indent=2))
            continue

        print(f"[{i}/{len(DOLCI_THINK_SOURCES)}] querying {slug!r} ({len(values)} dataset_source value(s)) ...",
              end=" ", flush=True)
        in_list = ", ".join("'" + v.replace("'", "''") + "'" for v in values)

        try:
            (num_rows,) = con.execute(
                f"SELECT count(*) FROM read_parquet('{glob}') WHERE dataset_source IN ({in_list})"
            ).fetchone()
            if num_rows == 0:
                status[slug] = "no_rows"
                print("no_rows")
                continue

            con.execute("SELECT setseed(?)", [SEED / 2147483647])
            rows_raw = con.execute(f"""
                SELECT id, messages, dataset_source
                FROM read_parquet('{glob}')
                WHERE dataset_source IN ({in_list})
                ORDER BY random()
                LIMIT {min(16, num_rows)}
            """).fetchall()

            rows = [
                truncate({"id": r[0], "messages": r[1], "dataset_source": r[2]})
                for r in rows_raw
            ]
            result = {
                "status": "ok",
                "config": "default",
                "split": "train",
                "features": [
                    {"feature_idx": 0, "name": "id", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 1, "name": "messages", "type": {"_type": "List"}},
                    {"feature_idx": 2, "name": "dataset_source", "type": {"dtype": "string", "_type": "Value"}},
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
    print(f"\nDone. {ok}/{len(DOLCI_THINK_SOURCES)} sources sampled successfully.")


if __name__ == "__main__":
    main()
