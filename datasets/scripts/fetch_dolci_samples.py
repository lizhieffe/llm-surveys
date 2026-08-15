"""Fetch 16 randomly-sampled rows for each dataset in Ai2's Dolci post-training
mixture for Olmo 3.

Dolci is Ai2's post-training data suite for Olmo 3 (and Olmo 3.1): each model
variant (Instruct, Think, RL-Zero) has its own SFT / DPO / RL dataset. This
currently covers just the Instruct-SFT stage
(https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT), which trains the
Olmo 3 Instruct models' supervised fine-tuning stage
(https://huggingface.co/allenai/Olmo-3.1-32B-Instruct). Add more entries to
CATEGORIES below (Dolci-Instruct-DPO, Dolci-Instruct-RL, the Think/RL-Zero
variants, ...) to extend this the same way HELMET/MMLongBench are extended.

Unlike HELMET/MMLongBench (hand-mapped to third-party mirrors of a tarball
release), every Dolci dataset has its own first-party, viewer-enabled repo
under allenai/, so this only needs the datasets-server rows API -- no mirror
hunting required.
"""

import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "dolci-samples"
NUM_EXAMPLES = 16
TRUNCATE_CHARS = 4000
TRUNCATE_LIST_ITEMS = 5
SERVER = "https://datasets-server.huggingface.co"
SEED = 42

CATEGORIES = [
    {
        "title": "Olmo 3 Instruct — SFT",
        "description": (
            "The supervised fine-tuning mixture behind the Olmo 3 Instruct models' first "
            "post-training stage (SFT → DPO → RLVR). 2,152,112 examples blending "
            "existing instruction datasets (FLAN v2, Tulu 3 personas, Aya, Evol-CodeAlpaca, "
            "WildGuardMix/WildJailbreak safety data, OpenThoughts 3, ...) with newly created "
            "Dolci sources (precise instruction-following, Python algorithms, upgraded WildChat "
            "responses, logic puzzles, verifiable reasoning, tool use)."
        ),
        "datasets": [
            {
                "name": "Dolci-Instruct-SFT",
                "description": (
                    "Olmo 3 Instruct's SFT-stage training data — 2.15M instruction-following "
                    "examples across 70+ languages, math, code, safety, reasoning, and tool use."
                ),
                "repo_id": "allenai/Dolci-Instruct-SFT",
            },
        ],
    },
]


def _get(url: str, timeout: int = 60, max_retries: int = 5) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-surveys"})
    backoff = 3.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise


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


def safe_filename(repo_id: str) -> str:
    return repo_id.replace("/", "__") + ".json"


def fetch_sample(repo_id: str, rng: random.Random) -> dict:
    try:
        size_resp = _get(f"{SERVER}/size?dataset={repo_id}")
        splits_resp = _get(f"{SERVER}/splits?dataset={repo_id}")
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)[:300]}

    splits = splits_resp.get("splits", [])
    if not splits:
        return {"status": "no_viewer", "error": "no splits returned"}
    chosen = next((s for s in splits if s["split"] == "train"), splits[0])
    config, split = chosen["config"], chosen["split"]
    num_rows = size_resp["size"]["dataset"]["num_rows"]

    n = min(NUM_EXAMPLES, num_rows)
    indices = sorted(rng.sample(range(num_rows), n))

    rows, features = [], None
    for idx in indices:
        try:
            resp = _get(f"{SERVER}/rows?dataset={repo_id}&config={config}&split={split}&offset={idx}&length=1")
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": str(e)[:300]}
        if features is None:
            features = resp.get("features", [])
        for r in resp.get("rows", []):
            rows.append(truncate(r["row"]))
        time.sleep(0.3)

    return {"status": "ok", "config": config, "split": split, "features": features, "rows": rows,
            "num_rows_total": num_rows}


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "dolci_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    all_repos = [d["repo_id"] for cat in CATEGORIES for d in cat["datasets"]]
    for i, repo_id in enumerate(all_repos, 1):
        out_path = SAMPLES_DIR / safe_filename(repo_id)
        if out_path.exists() and status.get(repo_id) == "ok":
            print(f"[{i}/{len(all_repos)}] skip (cached) {repo_id}")
            continue

        print(f"[{i}/{len(all_repos)}] fetching {repo_id} ...", end=" ", flush=True)
        rng = random.Random(SEED)  # fixed seed per dataset for reproducible sampling
        result = fetch_sample(repo_id, rng)
        status[repo_id] = result["status"]
        print(result["status"])
        if result["status"] == "ok":
            out_path.write_text(json.dumps(result, indent=2, default=str))
        else:
            print(f"    -> {result.get('error', '')}")
        status_path.write_text(json.dumps(status, indent=2))
        time.sleep(0.5)

    ok = sum(1 for v in status.values() if v == "ok")
    print(f"\nDone. {ok}/{len(all_repos)} datasets sampled successfully.")


if __name__ == "__main__":
    main()
