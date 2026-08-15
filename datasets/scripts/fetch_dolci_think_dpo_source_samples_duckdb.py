"""Sibling of fetch_dolci_think_source_samples_duckdb.py for
allenai/Dolci-Think-DPO-7B (the preference-tuning mixture behind Olmo 3
Think 7B's DPO stage) -- same idea (one category per upstream sub-source,
16 sampled rows each, via local DuckDB against downloaded parquet shards),
but simpler than either SFT dataset:

- Much smaller: 150,000 rows / 7 parquet shards / ~1.4GB, vs. millions of
  rows and tens of GB for the SFT mixtures.
- Row shape is preference-pair, not single-response: `prompt` (string),
  `chosen` / `rejected` (each a list of {content, role} messages),
  `chosen_model` / `rejected_model`, `dataset_source`, `id`,
  `preference_type` (uniformly "delta_learning" across the whole dataset --
  not useful as a category axis, see Geng et al. 2025,
  https://arxiv.org/abs/2507.06187, "Delta Learning", for the preference
  heuristic).
- Unlike the SFT cards, the Dolci-Think-DPO-7B card gives no prose
  breakdown of upstream sources at all -- just a one-line description
  ("150,000 preference pairs created with the Delta Learning heuristic").
  The DOLCI_THINK_DPO_SOURCES breakdown below comes entirely from a full
  local `GROUP BY dataset_source` (24 unique values, summing exactly to
  150,000 -- no missing/unmatched rows this time, unlike Think-SFT-7B's
  "Olmo Identity Prompts"), cross-referenced against Hugging Face search to
  find each raw dataset_source string's likely upstream repo (several
  dataset_source values are themselves the tail end of an
  ai2-adapt-dev/saumyamalik/jacobmorrison processing-pipeline repo name, so
  they were looked up directly rather than guessed).
"""

import json
import os
import urllib.request
from pathlib import Path

import duckdb

DATASET = "allenai/Dolci-Think-DPO-7B"
HF_TOKEN = os.environ.get("HF_TOKEN")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "dolci-samples"
NUM_SHARDS = 7
CACHE_DIR = Path(
    os.environ.get(
        "DOLCI_THINK_DPO_PARQUET_CACHE",
        "/tmp/claude-1000/-home-lizhi-developments-cs336-assignment5-alignment/"
        "33a4e6bc-871f-4837-afe1-6e9a29ff0efe/scratchpad/dolci-think-dpo-parquet",
    )
)
SEED = 42
TRUNCATE_CHARS = 4000
TRUNCATE_LIST_ITEMS = 5

# Each entry: slug, display title, the raw `dataset_source` value to match,
# a repo_id to link to (the upstream source repo where a clear one was
# found via HF search, else this dataset itself), license (Dolci-native
# entries are ODC-BY-1.0 per the card; others use the upstream repo's own
# license where known), and the verified live row count for this
# dataset_source value (there's no card breakdown to cross-check against
# here, so these counts *are* the primary source, not a secondary check).
DOLCI_THINK_DPO_SOURCES = [
    {
        "slug": "ultrafeedback-cleaned-dpo",
        "title": "UltraFeedback (cleaned)",
        "dataset_source_value": "ultrafeedback_cleaned_olmo2_7b",
        "repo_id": "allenai/ultrafeedback_binarized_cleaned",
        "license": "MIT",
        "row_count": 23202,
        "citation": None,
        "note": "Ai2's cleaned/binarized UltraFeedback, as used for OLMo 2 preference tuning.",
    },
    {
        "slug": "wildchat-filtered-sample-dpo",
        "title": "WildChat (filtered sample)",
        "dataset_source_value": "filtered_wc_sample_500k",
        "repo_id": "allenai/WildChat-1M",
        "license": "ODC-BY-1.0",
        "row_count": 17596,
        "citation": None,
        "note": None,
    },
    {
        "slug": "openthoughts-3-science-dpo",
        "title": "OpenThoughts 3 (science)",
        "dataset_source_value": "OpenThoughts3-full-filtered-science-no-cot",
        "repo_id": "open-thoughts/OpenThoughts3-1.2M",
        "license": "Apache 2.0",
        "row_count": 14967,
        "citation": None,
        "note": None,
    },
    {
        "slug": "flan-v2-dpo",
        "title": "FLAN v2",
        "dataset_source_value": "flan_v2_converted",
        "repo_id": "ai2-adapt-dev/flan_v2_converted",
        "license": None,
        "row_count": 14057,
        "citation": None,
        "note": None,
    },
    {
        "slug": "wildchat-gpt41-english-dpo",
        "title": "WildChat (GPT-4.1 regenerated, English)",
        "dataset_source_value": "Wildchat-1M-gpt-4.1-regenerated-english",
        "repo_id": "allenai/WildChat-1M",
        "license": "ODC-BY-1.0",
        "row_count": 13955,
        "citation": None,
        "note": None,
    },
    {
        "slug": "precise-if-verified-dpo",
        "title": "Precise IF (verified reasoning)",
        "dataset_source_value": "valpy_if_qwq_reasoning_verified_no_reasoning",
        "repo_id": "jacobmorrison/valpy_if_qwq_reasoning_verified_no_reasoning",
        "license": "ODC-BY-1.0",
        "row_count": 11101,
        "citation": "Pyatkin et al., 2025 (arXiv:2507.02833)",
        "note": "New multi-constraint instruction-following data building off “Generalizing Verifiable Instruction Following.”",
    },
    {
        "slug": "dolci-python-dpo",
        "title": "Dolci Python",
        "dataset_source_value": "correct-python-sft-187k",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "row_count": 9005,
        "citation": None,
        "note": "New prompts authored for Dolci; no separate upstream repo.",
    },
    {
        "slug": "tulu-3-persona-math-dpo",
        "title": "Tulu 3 Persona MATH",
        "dataset_source_value": "tulu-3-sft-personas-math",
        "repo_id": "allenai/tulu-3-sft-personas-math",
        "license": "ODC-BY-1.0",
        "row_count": 7252,
        "citation": None,
        "note": None,
    },
    {
        "slug": "wildchat-gpt41-non-english-dpo",
        "title": "WildChat (GPT-4.1 regenerated, non-English)",
        "dataset_source_value": "Wildchat-1m-gpt-4.1-regeneration-not-english",
        "repo_id": "allenai/WildChat-1M",
        "license": "ODC-BY-1.0",
        "row_count": 5220,
        "citation": None,
        "note": None,
    },
    {
        "slug": "evol-codealpaca-dpo",
        "title": "Evol CodeAlpaca",
        "dataset_source_value": "evol_codealpaca_heval_decontaminated",
        "repo_id": "theblackcat102/evol-codealpaca-v1",
        "license": "Apache 2.0",
        "row_count": 5171,
        "citation": "Luo et al., 2023",
        "note": None,
    },
    {
        "slug": "wildjailbreak-dpo",
        "title": "WildJailbreak",
        "dataset_source_value": "tulu_v3.9_wildjailbreak_decontaminated_50k",
        "repo_id": "allenai/wildjailbreak",
        "license": "ODC-BY-1.0",
        "row_count": 3885,
        "citation": "Wildteaming, 2024",
        "note": None,
    },
    {
        "slug": "wildguardmix-dpo",
        "title": "WildGuardMix",
        "dataset_source_value": "tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
        "repo_id": "allenai/wildguardmix",
        "license": "Apache 2.0",
        "row_count": 3884,
        "citation": "Han et al., 2024",
        "note": None,
    },
    {
        "slug": "aya-dpo",
        "title": "Aya",
        "dataset_source_value": "tulu_v3.9_aya_100k",
        "repo_id": "CohereForAI/aya_dataset",
        "license": "Apache 2.0",
        "row_count": 2863,
        "citation": "Singh et al., 2024",
        "note": None,
    },
    {
        "slug": "openmathinstruct-2-dpo",
        "title": "OpenMathInstruct 2",
        "dataset_source_value": "tulu_v3.9_open_math_2_gsm8k_50k",
        "repo_id": "nvidia/OpenMathInstruct-2",
        "license": "CC-BY-4.0",
        "row_count": 2561,
        "citation": None,
        "note": None,
    },
    {
        "slug": "tulu-3-persona-gsm-dpo",
        "title": "Tulu 3 Persona GSM",
        "dataset_source_value": "tulu-3-sft-personas-math-grade",
        "repo_id": "allenai/tulu-3-sft-personas-math-grade",
        "license": "ODC-BY-1.0",
        "row_count": 2520,
        "citation": None,
        "note": None,
    },
    {
        "slug": "if-sft-verified-permissive-dpo",
        "title": "IF SFT Data (verified, permissive)",
        "dataset_source_value": "IF_sft_data_verified_permissive",
        "repo_id": "allenai/IF_sft_data_verified",
        "license": None,
        "row_count": 2482,
        "citation": None,
        "note": "Permissively-licensed subset of Ai2's verified instruction-following SFT data.",
    },
    {
        "slug": "tulu-3-persona-if-o3-dpo",
        "title": "Tulu 3 Persona IF (o3)",
        "dataset_source_value": "tulu-3-sft-personas-instruction-following-o3",
        "repo_id": "allenai/tulu-3-sft-personas-instruction-following",
        "license": "ODC-BY-1.0",
        "row_count": 2447,
        "citation": None,
        "note": "Generated with o3.",
    },
    {
        "slug": "tulu-3-persona-python-dpo",
        "title": "Tulu 3 Persona Python",
        "dataset_source_value": "personahub_code_v2_34999",
        "repo_id": "allenai/tulu-3-sft-personas-code",
        "license": "ODC-BY-1.0",
        "row_count": 1736,
        "citation": None,
        "note": None,
    },
    {
        "slug": "sciriff-dpo",
        "title": "SciRIFF",
        "dataset_source_value": "tulu_v3.9_sciriff_10k",
        "repo_id": "allenai/SciRIFF",
        "license": "ODC-BY-1.0",
        "row_count": 1554,
        "citation": "Wadden et al., 2024",
        "note": None,
    },
    {
        "slug": "openassistant-guanaco-dpo",
        "title": "OpenAssistant Guanaco",
        "dataset_source_value": "oasst1_converted",
        "repo_id": "timdettmers/openassistant-guanaco",
        "license": "Apache 2.0",
        "row_count": 1259,
        "citation": "Kopf et al., 2024",
        "note": None,
    },
    {
        "slug": "tulu-3-persona-algebra-dpo",
        "title": "Tulu 3 Persona Algebra",
        "dataset_source_value": "tulu-3-sft-personas-algebra",
        "repo_id": "allenai/tulu-3-sft-personas-algebra",
        "license": "ODC-BY-1.0",
        "row_count": 930,
        "citation": None,
        "note": None,
    },
    {
        "slug": "tablegpt-dpo",
        "title": "TableGPT",
        "dataset_source_value": "tulu_v3.9_table_gpt_5k",
        "repo_id": "LipengCS/Table-GPT",
        "license": "MIT",
        "row_count": 834,
        "citation": "Zha et al., 2023",
        "note": None,
    },
    {
        "slug": "coconot-regenerated-dpo",
        "title": "CoCoNot (regenerated)",
        "dataset_source_value": "tulu-3-sft-coconot-regenerated",
        "repo_id": "allenai/coconot",
        "license": "ODC-BY-1.0",
        "row_count": 790,
        "citation": "Brahman et al., 2024",
        "note": None,
    },
    {
        "slug": "daringanteater-dpo",
        "title": "DaringAnteater",
        "dataset_source_value": "DaringAnteater-prefs_olmo2_7b",
        "repo_id": "nvidia/Daring-Anteater",
        "license": "CC-BY-4.0",
        "row_count": 729,
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
    return f"dolci-think-dpo-source__{slug}.json"


def main() -> None:
    download_shards()
    glob = str(CACHE_DIR / "*.parquet")

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "dolci_think_dpo_source_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    for i, src in enumerate(DOLCI_THINK_DPO_SOURCES, 1):
        slug = src["slug"]
        out_path = SAMPLES_DIR / safe_filename(slug)
        if out_path.exists() and status.get(slug) == "ok":
            print(f"[{i}/{len(DOLCI_THINK_DPO_SOURCES)}] skip (cached) {slug}")
            continue

        value = src["dataset_source_value"].replace("'", "''")
        print(f"[{i}/{len(DOLCI_THINK_DPO_SOURCES)}] querying {slug!r} ...", end=" ", flush=True)

        try:
            (num_rows,) = con.execute(
                f"SELECT count(*) FROM read_parquet('{glob}') WHERE dataset_source = '{value}'"
            ).fetchone()
            if num_rows == 0:
                status[slug] = "no_rows"
                print("no_rows")
                continue

            con.execute("SELECT setseed(?)", [SEED / 2147483647])
            rows_raw = con.execute(f"""
                SELECT id, prompt, chosen, rejected, chosen_model, rejected_model, dataset_source
                FROM read_parquet('{glob}')
                WHERE dataset_source = '{value}'
                ORDER BY random()
                LIMIT {min(16, num_rows)}
            """).fetchall()

            rows = [
                truncate({
                    "id": r[0],
                    "prompt": r[1],
                    "chosen": r[2],
                    "rejected": r[3],
                    "chosen_model": r[4],
                    "rejected_model": r[5],
                    "dataset_source": r[6],
                })
                for r in rows_raw
            ]
            result = {
                "status": "ok",
                "config": "default",
                "split": "train",
                "features": [
                    {"feature_idx": 0, "name": "id", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 1, "name": "prompt", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 2, "name": "chosen", "type": {"_type": "List"}},
                    {"feature_idx": 3, "name": "rejected", "type": {"_type": "List"}},
                    {"feature_idx": 4, "name": "chosen_model", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 5, "name": "rejected_model", "type": {"dtype": "string", "_type": "Value"}},
                    {"feature_idx": 6, "name": "dataset_source", "type": {"dtype": "string", "_type": "Value"}},
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
    print(f"\nDone. {ok}/{len(DOLCI_THINK_DPO_SOURCES)} sources sampled successfully.")


if __name__ == "__main__":
    main()
