"""Combine categories.json + dataset_metadata.json + sample_status.json into a
single data/manifest.json for the frontend to consume.

Full 16-row samples stay in their own per-dataset files under data/samples/ and
are fetched lazily by the page, so the manifest itself stays small.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    categories = json.loads((DATA_DIR / "categories.json").read_text())
    dataset_metadata = json.loads((DATA_DIR / "dataset_metadata.json").read_text())
    sample_status = json.loads((DATA_DIR / "sample_status.json").read_text())

    out_categories = []
    for cat in categories:
        datasets = []
        for repo_id in cat["datasets"]:
            meta = dataset_metadata.get(repo_id, {})
            status = sample_status.get(repo_id, "missing")
            datasets.append(
                {
                    "repo_id": repo_id,
                    "url": meta.get("url", f"https://huggingface.co/datasets/{repo_id}"),
                    "description": meta.get("description", ""),
                    "downloads": meta.get("downloads", 0),
                    "likes": meta.get("likes", 0),
                    "license": meta.get("license"),
                    "gated": meta.get("gated", False),
                    "sample_status": status,
                    "sample_file": f"data/samples/{repo_id.replace('/', '__')}.json" if status == "ok" else None,
                }
            )
        out_categories.append(
            {
                "slug": cat["slug"],
                "title": cat["title"],
                "description": cat["description"],
                "url": cat["url"],
                "datasets": datasets,
            }
        )

    unique_ids = {d for cat in categories for d in cat["datasets"]}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://huggingface.co/nvidia/collections?search=nemotron",
        "num_categories": len(out_categories),
        "num_unique_datasets": len(unique_ids),
        "num_sampled_ok": sum(1 for v in sample_status.values() if v == "ok"),
        "categories": out_categories,
    }

    out_path = DATA_DIR / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_path}: {manifest['num_categories']} categories, "
          f"{manifest['num_unique_datasets']} unique datasets, "
          f"{manifest['num_sampled_ok']} sampled ok")


if __name__ == "__main__":
    main()
