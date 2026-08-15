"""Build datasets/data/dolci-source-manifest.json: one category per
Dolci-Instruct-SFT sub-source (see SOURCES in fetch_dolci_source_samples.py),
each carrying its own 16 sampled rows filtered by `source_dataset`.

This is the finer-grained sibling of build_dolci_manifest.py, which treats
the whole Dolci-Instruct-SFT repo as a single category.
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fetch_dolci_source_samples import DATASET, HF_TOKEN, SOURCES, safe_filename

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOLCI_INSTRUCT_MODEL = "https://huggingface.co/allenai/Olmo-3.1-32B-Instruct"


def _get(url: str) -> dict:
    headers = {"User-Agent": "llm-surveys"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


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
    sample_status = json.loads((DATA_DIR / "dolci_source_sample_status.json").read_text())
    meta_cache_path = DATA_DIR / "dolci_source_dataset_metadata.json"
    meta_cache = json.loads(meta_cache_path.read_text()) if meta_cache_path.exists() else {}

    out_categories = []
    for src in SOURCES:
        repo_id = src["repo_id"]
        if repo_id not in meta_cache:
            print("fetching metadata:", repo_id)
            meta_cache[repo_id] = fetch_metadata(repo_id)
            time.sleep(0.3)
        meta = meta_cache[repo_id]

        slug = src["slug"]
        status = sample_status.get(slug, "missing")

        desc_bits = []
        if src.get("note"):
            desc_bits.append(src["note"])
        if src.get("citation"):
            desc_bits.append(f"({src['citation']})")
        description = " ".join(desc_bits) if desc_bits else None

        is_dolci_native = repo_id == DATASET
        card = {
            "name": src["title"],
            "repo_id": repo_id,
            "url": f"https://huggingface.co/datasets/{repo_id}",
            "description": description,
            "metric": f"{src['card_count']:,} prompts",
            # Prefer the license as stated on the Dolci-Instruct-SFT card
            # (per-source) over the upstream repo's HF license tag, which is
            # sometimes missing or generic; fall back to the fetched tag.
            "license": src.get("license") or meta.get("license"),
            "downloads": None if is_dolci_native else meta.get("downloads", 0),
            "likes": None if is_dolci_native else meta.get("likes", 0),
            "gated": meta.get("gated", False),
            "sample_status": status,
            "sample_file": f"../data/dolci-samples/{safe_filename(slug)}" if status == "ok" else None,
        }

        out_categories.append({
            "slug": slug,
            "title": src["title"],
            "description": None,
            "url": card["url"],
            "datasets": [card],
        })

    meta_cache_path.write_text(json.dumps(meta_cache, indent=2))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT",
        "source_model": DOLCI_INSTRUCT_MODEL,
        "num_categories": len(out_categories),
        "num_unique_datasets": len(out_categories),
        "num_sampled_ok": sum(1 for v in sample_status.values() if v == "ok"),
        "categories": out_categories,
    }

    out_path = DATA_DIR / "dolci-source-manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_path}: {manifest['num_categories']} categories, "
          f"{manifest['num_sampled_ok']} sampled ok")


if __name__ == "__main__":
    main()
