"""Fetch lightweight metadata (license, downloads, likes, short description) for
every unique dataset referenced in data/categories.json.

Writes data/dataset_metadata.json keyed by repo_id.
"""

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-dataset-explorer"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def short_description(raw: str, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", raw or "").strip()
    text = re.sub(r"See the full description.*$", "", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text


def main() -> None:
    categories = json.loads((DATA_DIR / "categories.json").read_text())
    unique_datasets = sorted({d for cat in categories for d in cat["datasets"]})

    out_path = DATA_DIR / "dataset_metadata.json"
    metadata = json.loads(out_path.read_text()) if out_path.exists() else {}

    for i, repo_id in enumerate(unique_datasets, 1):
        if repo_id in metadata:
            print(f"[{i}/{len(unique_datasets)}] skip (cached) {repo_id}")
            continue
        print(f"[{i}/{len(unique_datasets)}] fetching {repo_id} ...", end=" ", flush=True)
        try:
            data = _get(f"https://huggingface.co/api/datasets/{repo_id}")
            license_tag = next(
                (t.split(":", 1)[1] for t in data.get("tags", []) if t.startswith("license:")),
                None,
            )
            metadata[repo_id] = {
                "description": short_description(data.get("description", "")),
                "downloads": data.get("downloads", 0),
                "likes": data.get("likes", 0),
                "gated": bool(data.get("gated", False)),
                "license": license_tag,
                "lastModified": data.get("lastModified"),
                "url": f"https://huggingface.co/datasets/{repo_id}",
            }
            print("ok")
        except urllib.error.HTTPError as e:
            metadata[repo_id] = {"error": f"HTTP {e.code}", "url": f"https://huggingface.co/datasets/{repo_id}"}
            print(f"HTTP {e.code}")
        except Exception as e:  # noqa: BLE001
            metadata[repo_id] = {"error": str(e)[:200], "url": f"https://huggingface.co/datasets/{repo_id}"}
            print(f"error: {e}")

        out_path.write_text(json.dumps(metadata, indent=2))
        time.sleep(0.3)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
