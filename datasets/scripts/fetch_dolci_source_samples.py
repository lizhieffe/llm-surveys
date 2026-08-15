"""Fetch 16 randomly-sampled rows for each *sub-source* inside Ai2's
Dolci-Instruct-SFT mixture.

Dolci-Instruct-SFT (https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT)
is a 2.15M-row blend of ~20 existing instruction datasets plus several newly
created Dolci sources. Every row carries a `source_dataset` string column
identifying which of those ~20 sources it came from, and the datasets-server
`/filter` endpoint (SQL-ish WHERE clause, column names must be double-quoted)
lets us query rows for one source at a time:

    GET /filter?dataset=...&where="source_dataset"='Tulu 3 Persona MATH'&offset=0&length=1

This script uses that to sample each SOURCES entry below independently, the
same way fetch_dolci_samples.py samples whole repos -- but here all 22
sources live inside the single Dolci-Instruct-SFT repo, filtered by value
rather than fetched from 22 separate repos.

The one exception found empirically: the real data contains an
"OpenMathInstruct 2" source (50,000 rows) that isn't mentioned in the
Dolci-Instruct-SFT dataset card's prose breakdown. It's included here because
the 22 source_dataset values sum exactly to the card's stated 2,152,112 total,
confirming it's real and simply omitted from the card text.
"""

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "dolci-samples"
NUM_EXAMPLES = 16
TRUNCATE_CHARS = 4000
TRUNCATE_LIST_ITEMS = 5
SERVER = "https://datasets-server.huggingface.co"
DATASET = "allenai/Dolci-Instruct-SFT"
CONFIG = "default"
SPLIT = "train"
SEED = 42

# Each entry: slug, display title, the exact `source_dataset` value to filter
# on, the license/citation/repo as given by the Dolci-Instruct-SFT card (or
# left None for sources newly authored for Dolci, which have no separate
# upstream repo), and the prompt count the card states (may differ slightly
# from the live count if the card was written against an earlier snapshot).
SOURCES = [
    # --- existing prompts, blended into Dolci -----------------------------
    {
        "slug": "openthoughts-3",
        "title": "OpenThoughts 3",
        "source_dataset": "Dolci Instruct OpenThoughts3+ Science",
        "repo_id": "open-thoughts/OpenThoughts3-1.2M",
        "license": "Apache 2.0",
        "card_count": 99268,
        "citation": None,
        "note": (
            "Extended to 32K context length and downsampled code prompts to 16x "
            "multiple (941,166 total prompts upstream); reasoning traces removed "
            "for the instruct variant used here."
        ),
    },
    {
        "slug": "coconot",
        "title": "CoCoNot",
        "source_dataset": "CoCoNot",
        "repo_id": "allenai/coconot",
        "license": "ODC-BY-1.0",
        "card_count": 10957,
        "citation": "Brahman et al., 2024",
    },
    {
        "slug": "flan-v2",
        "title": "FLAN v2",
        "source_dataset": "FLAN",
        "repo_id": "ai2-adapt-dev/flan_v2_converted",
        "license": None,
        "card_count": 89981,
        "citation": "Longpre et al., 2023",
    },
    {
        "slug": "openassistant-guanaco",
        "title": "OpenAssistant Guanaco",
        "source_dataset": "OpenAssistant",
        "repo_id": "timdettmers/openassistant-guanaco",
        "license": "Apache 2.0",
        "card_count": 7132,
        "citation": "Kopf et al., 2024",
    },
    {
        "slug": "tulu-3-persona-math",
        "title": "Tulu 3 Persona MATH",
        "source_dataset": "Tulu 3 Persona MATH",
        "repo_id": "allenai/tulu-3-sft-personas-math",
        "license": "ODC-BY-1.0",
        "card_count": 149958,
        "citation": None,
    },
    {
        "slug": "tulu-3-persona-gsm",
        "title": "Tulu 3 Persona GSM",
        "source_dataset": "Tulu 3 Persona GSM",
        "repo_id": "allenai/tulu-3-sft-personas-math-grade",
        "license": "ODC-BY-1.0",
        "card_count": 49980,
        "citation": None,
    },
    {
        "slug": "tulu-3-persona-python",
        "title": "Tulu 3 Persona Python",
        "source_dataset": "Tulu 3 Persona Python",
        "repo_id": "allenai/tulu-3-sft-personas-code",
        "license": "ODC-BY-1.0",
        "card_count": 34999,
        "citation": None,
    },
    {
        "slug": "tulu-3-persona-algebra",
        "title": "Tulu 3 Persona Algebra",
        "source_dataset": "Tulu 3 Persona Algebra",
        "repo_id": "allenai/tulu-3-sft-personas-algebra",
        "license": "ODC-BY-1.0",
        "card_count": 19999,
        "citation": None,
    },
    {
        "slug": "wildguardmix",
        "title": "Tulu 3 WildGuardMix",
        "source_dataset": "WildGuardMix",
        "repo_id": "allenai/wildguardmix",
        "license": "Apache 2.0",
        "card_count": 49373,
        "citation": "Han et al., 2024",
    },
    {
        "slug": "wildjailbreak",
        "title": "Tulu 3 WildJailbreak",
        "source_dataset": "WildJailbreak",
        "repo_id": "allenai/wildjailbreak",
        "license": "ODC-BY-1.0",
        "card_count": 49965,
        "citation": "Wildteaming, 2024",
    },
    {
        "slug": "aya",
        "title": "Aya",
        "source_dataset": "Aya",
        "repo_id": "CohereForAI/aya_dataset",
        "license": "Apache 2.0",
        "card_count": 99987,
        "citation": "Singh et al., 2024",
    },
    {
        "slug": "tablegpt",
        "title": "TableGPT",
        "source_dataset": "TableGPT",
        "repo_id": "LipengCS/Table-GPT",
        "license": "MIT",
        "card_count": 5000,
        "citation": "Zha et al., 2023",
    },
    {
        "slug": "sciriff",
        "title": "SciRIFF",
        "source_dataset": "SciRiff",
        "repo_id": "allenai/SciRIFF",
        "license": "ODC-BY-1.0",
        "card_count": 4557,
        "citation": "Wadden et al., 2024",
    },
    {
        "slug": "evol-codealpaca",
        "title": "Evol CodeAlpaca",
        "source_dataset": "Evol CodeAlpaca",
        "repo_id": "theblackcat102/evol-codealpaca-v1",
        "license": "Apache 2.0",
        "card_count": 107270,
        "citation": "Luo et al., 2023",
    },
    {
        "slug": "openmathinstruct-2",
        "title": "OpenMathInstruct 2",
        "source_dataset": "OpenMathInstruct 2",
        "repo_id": "nvidia/OpenMathInstruct-2",
        "license": "CC-BY-4.0",
        "card_count": 50000,
        "citation": None,
        "note": (
            "Present in the live data (50,000 rows) but not called out in the "
            "Dolci-Instruct-SFT card's prose breakdown -- included here because "
            "the 22 source_dataset values sum exactly to the card's stated "
            "2,152,112 total."
        ),
    },
    # --- new prompts authored for Dolci ------------------------------------
    {
        "slug": "wildchat-upgraded",
        "title": "WildChat (upgraded w/ GPT-4.1)",
        "source_dataset": "Wildchat",
        "repo_id": "allenai/WildChat-1M",
        "license": "ODC-BY-1.0",
        "card_count": 302406,
        "citation": "Zhao et al., 2024",
        "note": "Original WildChat prompts, re-answered with GPT-4.1 for Dolci.",
    },
    {
        "slug": "dolci-precise-if",
        "title": "Dolci Tülu 3 Precise IF",
        "source_dataset": "Dolci Instruct Precise IF",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 136833,
        "citation": None,
        "note": "New prompts authored for Dolci; no separate upstream repo.",
    },
    {
        "slug": "dolci-python-algorithms",
        "title": "Dolci Instruct Python Algorithms",
        "source_dataset": "Dolci Instruct Python Algorithms",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 186345,
        "citation": None,
        "note": "New prompts authored for Dolci; no separate upstream repo.",
    },
    {
        "slug": "logic-puzzles",
        "title": "Logic Puzzles",
        "source_dataset": "Logic Puzzles",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 159882,
        "citation": None,
        "note": "New prompts authored for Dolci; no separate upstream repo.",
    },
    {
        "slug": "verifiable-reasoning",
        "title": "Verifiable Reasoning",
        "source_dataset": "Verifiable Reasoning",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 310572,
        "citation": None,
        "note": "New prompts authored for Dolci; no separate upstream repo.",
    },
    {
        "slug": "hardcoded-data",
        "title": "New Hardcoded Data",
        "source_dataset": "Hardcoded Data",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 69,
        "citation": None,
        "note": "New prompts authored for Dolci; no separate upstream repo.",
    },
    {
        "slug": "dolci-tool-use",
        "title": "Dolci Instruct Tool Use",
        "source_dataset": "Dolci Instruct Tool Use",
        "repo_id": DATASET,
        "license": "ODC-BY-1.0",
        "card_count": 227579,
        "citation": None,
        "note": "New prompts authored for Dolci; no separate upstream repo.",
    },
]


def _get(url: str, timeout: int = 60, max_retries: int = 12) -> dict:
    headers = {"User-Agent": "llm-surveys"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    backoff = 4.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            # datasets-server's filter endpoint is flaky while its DuckDB
            # index warms up across worker replicas: it returns 500 with
            # "index is loading" on some requests and a generic 500
            # "Unexpected error" on others for the same underlying cause.
            # Treat all 429/5xx as transient and retry with backoff.
            transient = e.code == 429 or e.code >= 500
            if transient and attempt < max_retries - 1:
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)
                continue
            raise urllib.error.HTTPError(e.url, e.code, body, e.headers, None)


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
    return f"dolci-source__{slug}.json"


def filter_query(source_dataset: str, offset: int, length: int) -> dict:
    value = source_dataset.replace("'", "''")
    where = f"\"source_dataset\"='{value}'"
    url = (
        f"{SERVER}/filter?dataset={urllib.parse.quote(DATASET)}&config={CONFIG}&split={SPLIT}"
        f"&where={urllib.parse.quote(where)}&offset={offset}&length={length}"
    )
    return _get(url)


def fetch_source_sample(source_dataset: str, rng: random.Random) -> dict:
    probe = filter_query(source_dataset, 0, 1)
    num_rows = probe.get("num_rows_total", 0)
    if num_rows == 0:
        return {"status": "no_rows", "error": "filter returned 0 rows"}

    n = min(NUM_EXAMPLES, num_rows)
    indices = sorted(rng.sample(range(num_rows), n))

    rows, features = [], None
    for idx in indices:
        resp = filter_query(source_dataset, idx, 1)
        if features is None:
            features = resp.get("features", [])
        for r in resp.get("rows", []):
            rows.append(truncate(r["row"]))
        time.sleep(0.25)

    return {
        "status": "ok",
        "config": CONFIG,
        "split": SPLIT,
        "features": features,
        "rows": rows,
        "num_rows_total": num_rows,
    }


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "dolci_source_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    for i, src in enumerate(SOURCES, 1):
        slug = src["slug"]
        out_path = SAMPLES_DIR / safe_filename(slug)
        if out_path.exists() and status.get(slug) == "ok":
            print(f"[{i}/{len(SOURCES)}] skip (cached) {slug}")
            continue

        print(f"[{i}/{len(SOURCES)}] fetching {slug!r} (source_dataset={src['source_dataset']!r}) ...",
              end=" ", flush=True)
        rng = random.Random(SEED)
        try:
            result = fetch_source_sample(src["source_dataset"], rng)
        except Exception as e:  # noqa: BLE001
            result = {"status": "error", "error": str(e)[:300]}
        status[slug] = result["status"]
        print(result["status"])
        if result["status"] == "ok":
            out_path.write_text(json.dumps(result, indent=2, default=str))
        else:
            print(f"    -> {result.get('error', '')}")
        status_path.write_text(json.dumps(status, indent=2))
        time.sleep(0.5)

    ok = sum(1 for v in status.values() if v == "ok")
    print(f"\nDone. {ok}/{len(SOURCES)} sources sampled successfully.")


if __name__ == "__main__":
    main()
