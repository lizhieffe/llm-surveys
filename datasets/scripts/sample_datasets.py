"""Sample 16 example rows from each dataset in data/categories.json.

Uses the Hugging Face datasets-server API (https://datasets-server.huggingface.co),
which serves rows out of a pre-built Parquet export -- so this works even for
multi-terabyte datasets like Nemotron-CC without downloading them.

Writes one JSON file per dataset to data/samples/<owner>__<name>.json, and a
data/sample_status.json summary (ok / gated / no_viewer / error) so failures are
easy to see and re-runs only retry what's missing.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "samples"
NUM_EXAMPLES = 16
SERVER = "https://datasets-server.huggingface.co"


def _get(url: str, timeout: int = 60, max_retries: int = 5) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-dataset-explorer"})
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


def safe_filename(repo_id: str) -> str:
    return repo_id.replace("/", "__") + ".json"


def fetch_sample(repo_id: str) -> dict:
    """Returns {"status": ..., ...}. status is one of: ok, gated, no_viewer, error."""
    try:
        splits_resp = _get(f"{SERVER}/splits?dataset={repo_id}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code in (401, 403) or "gated" in body.lower():
            return {"status": "gated", "error": body[:300]}
        return {"status": "no_viewer", "error": f"HTTP {e.code}: {body[:300]}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)[:300]}

    splits = splits_resp.get("splits", [])
    if not splits:
        return {"status": "no_viewer", "error": "no splits returned"}

    # Prefer a "train" split if present, else the first available.
    chosen = next((s for s in splits if s["split"] == "train"), splits[0])
    config, split = chosen["config"], chosen["split"]

    try:
        rows_resp = _get(
            f"{SERVER}/rows?dataset={repo_id}&config={config}&split={split}"
            f"&offset=0&length={NUM_EXAMPLES}"
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code in (401, 403):
            return {"status": "gated", "error": body[:300]}
        return {"status": "no_viewer", "error": f"HTTP {e.code}: {body[:300]}"}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)[:300]}

    return {
        "status": "ok",
        "config": config,
        "split": split,
        "features": rows_resp.get("features", []),
        "rows": [r["row"] for r in rows_resp.get("rows", [])],
    }


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    categories = json.loads((DATA_DIR / "categories.json").read_text())
    unique_datasets = sorted({d for cat in categories for d in cat["datasets"]})

    status_path = DATA_DIR / "sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    for i, repo_id in enumerate(unique_datasets, 1):
        out_path = SAMPLES_DIR / safe_filename(repo_id)
        if out_path.exists() and status.get(repo_id) == "ok":
            print(f"[{i}/{len(unique_datasets)}] skip (cached) {repo_id}")
            continue

        print(f"[{i}/{len(unique_datasets)}] fetching {repo_id} ...", end=" ", flush=True)
        result = fetch_sample(repo_id)
        status[repo_id] = result["status"]
        print(result["status"])

        if result["status"] == "ok":
            out_path.write_text(json.dumps(result, indent=2, default=str))
        else:
            print(f"    -> {result.get('error', '')}")

        status_path.write_text(json.dumps(status, indent=2))
        time.sleep(1.2)

    ok = sum(1 for v in status.values() if v == "ok")
    print(f"\nDone. {ok}/{len(unique_datasets)} datasets sampled successfully.")
    for s in ("gated", "no_viewer", "error"):
        failed = [k for k, v in status.items() if v == s]
        if failed:
            print(f"{s} ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
