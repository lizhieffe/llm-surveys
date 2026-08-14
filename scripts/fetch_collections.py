"""Fetch NVIDIA Nemotron dataset collections from the Hugging Face Hub API.

Reproduces what you'd see browsing https://huggingface.co/nvidia/collections?search=nemotron,
but via the structured API instead of scraping HTML.

Writes data/categories.json: a list of {slug, title, description, url, datasets: [repo_id, ...]}
for every nvidia collection whose title/slug mentions "nemotron" and whose items are
(at least in part) datasets.
"""

import json
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
API_BASE = "https://huggingface.co/api"


def _get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-dataset-explorer"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def list_nvidia_collections() -> list[dict]:
    """Page through every collection owned by nvidia."""
    collections = []
    url = f"{API_BASE}/collections?owner=nvidia&sort=trending&limit=100"
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-dataset-explorer"})
        with urllib.request.urlopen(req) as resp:
            collections.extend(json.loads(resp.read()))
            link = resp.headers.get("Link")
        url = None
        if link and 'rel="next"' in link:
            url = link.split(";")[0].strip("<> ")
        time.sleep(0.2)
    return collections


def fetch_collection_detail(slug: str) -> dict:
    return _get(f"{API_BASE}/collections/{slug}")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    all_collections = list_nvidia_collections()
    print(f"Found {len(all_collections)} total nvidia collections")

    nemotron_collections = [
        c
        for c in all_collections
        if "nemotron" in c["title"].lower()
        or "nemotron" in c["slug"].lower()
        or "nemotron" in (c.get("description") or "").lower()
    ]
    print(f"Found {len(nemotron_collections)} nemotron-related collections")

    categories = []
    for c in nemotron_collections:
        detail = fetch_collection_detail(c["slug"])
        dataset_items = [i["id"] for i in detail["items"] if i["type"] == "dataset"]
        if not dataset_items:
            # Skip collections that are purely models/spaces/papers (not dataset categories).
            continue
        categories.append(
            {
                "slug": c["slug"],
                "title": detail["title"],
                "description": detail.get("description") or "",
                "url": f"https://huggingface.co/collections/{c['slug']}",
                "datasets": dataset_items,
            }
        )
        print(f"  {detail['title']}: {len(dataset_items)} datasets")
        time.sleep(0.2)

    out_path = DATA_DIR / "categories.json"
    out_path.write_text(json.dumps(categories, indent=2))

    unique_datasets = sorted({d for cat in categories for d in cat["datasets"]})
    print(f"\n{len(categories)} dataset categories, {len(unique_datasets)} unique dataset repos")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
