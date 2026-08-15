"""Build datasets/data/dolci-manifest.json from the CATEGORIES table in
fetch_dolci_samples.py plus each repo's basic HF metadata and sample status.
"""

import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fetch_dolci_samples import CATEGORIES, safe_filename

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOLCI_INSTRUCT_MODEL = "https://huggingface.co/allenai/Olmo-3.1-32B-Instruct"
OLMO_CORE_REPO = "https://github.com/allenai/OLMo-core"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-surveys"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def short_description(raw: str, limit: int = 280) -> str:
    text = re.sub(r"\s+", " ", raw or "").strip()
    text = re.sub(r"See the full description.*$", "", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "..."
    return text


def fetch_metadata(repo_id: str) -> dict:
    try:
        data = _get(f"https://huggingface.co/api/datasets/{repo_id}")
        license_tag = next(
            (t.split(":", 1)[1] for t in data.get("tags", []) if t.startswith("license:")), None
        )
        return {
            "downloads": data.get("downloads", 0),
            "likes": data.get("likes", 0),
            "gated": bool(data.get("gated", False)),
            "license": license_tag,
        }
    except Exception:  # noqa: BLE001
        return {}


def main() -> None:
    sample_status = json.loads((DATA_DIR / "dolci_sample_status.json").read_text())
    meta_cache_path = DATA_DIR / "dolci_dataset_metadata.json"
    meta_cache = json.loads(meta_cache_path.read_text()) if meta_cache_path.exists() else {}

    out_categories = []
    for cat in CATEGORIES:
        datasets = []
        for d in cat["datasets"]:
            repo_id = d["repo_id"]
            if repo_id not in meta_cache:
                print("fetching metadata:", repo_id)
                meta_cache[repo_id] = fetch_metadata(repo_id)
                time.sleep(0.3)
            meta = meta_cache[repo_id]
            status = sample_status.get(repo_id, "missing")
            datasets.append({
                "name": d["name"],
                "repo_id": repo_id,
                "url": f"https://huggingface.co/datasets/{repo_id}",
                "description": short_description(d["description"]),
                "downloads": meta.get("downloads", 0),
                "likes": meta.get("likes", 0),
                "license": meta.get("license"),
                "gated": meta.get("gated", False),
                "sample_status": status,
                "sample_file": f"../data/dolci-samples/{safe_filename(repo_id)}" if status == "ok" else None,
            })
        out_categories.append({
            "slug": cat["title"].lower().replace(" ", "-").replace("—", "-"),
            "title": cat["title"],
            "description": cat["description"],
            "url": "https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT",
            "datasets": datasets,
        })

    meta_cache_path.write_text(json.dumps(meta_cache, indent=2))

    unique_ids = {d["repo_id"] for cat in CATEGORIES for d in cat["datasets"]}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT",
        "source_model": DOLCI_INSTRUCT_MODEL,
        "source_olmo_core": OLMO_CORE_REPO,
        "num_categories": len(out_categories),
        "num_unique_datasets": len(unique_ids),
        "num_sampled_ok": sum(1 for v in sample_status.values() if v == "ok"),
        "categories": out_categories,
    }

    out_path = DATA_DIR / "dolci-manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_path}: {manifest['num_categories']} categories, "
          f"{manifest['num_unique_datasets']} datasets, {manifest['num_sampled_ok']} sampled ok")


if __name__ == "__main__":
    main()
