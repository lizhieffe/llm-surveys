"""Build datasets/data/dolci-source-manifest.json: one category per
upstream sub-source, across all Dolci stages covered so far --
Dolci-Instruct-SFT (see SOURCES in fetch_dolci_source_samples.py),
Dolci-Think-SFT-7B (see DOLCI_THINK_SOURCES in
fetch_dolci_think_source_samples_duckdb.py), and Dolci-Think-DPO-7B (see
DOLCI_THINK_DPO_SOURCES in fetch_dolci_think_dpo_source_samples_duckdb.py)
-- each carrying its own 16 sampled rows filtered by that source's
identifying column.

This is the finer-grained sibling of build_dolci_manifest.py, which treats
a whole Dolci repo as a single category.
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fetch_dolci_source_samples import DATASET as INSTRUCT_DATASET
from fetch_dolci_source_samples import HF_TOKEN, SOURCES
from fetch_dolci_source_samples import safe_filename as instruct_safe_filename
from fetch_dolci_think_source_samples_duckdb import DATASET as THINK_DATASET
from fetch_dolci_think_source_samples_duckdb import DOLCI_THINK_SOURCES
from fetch_dolci_think_source_samples_duckdb import safe_filename as think_safe_filename
from fetch_dolci_think_dpo_source_samples_duckdb import DATASET as THINK_DPO_DATASET
from fetch_dolci_think_dpo_source_samples_duckdb import DOLCI_THINK_DPO_SOURCES
from fetch_dolci_think_dpo_source_samples_duckdb import safe_filename as think_dpo_safe_filename

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


def build_stage_categories(sources, native_dataset_repo, safe_filename_fn, status_path, title_prefix, meta_cache):
    """Build manifest categories for one Dolci stage (Instruct-SFT or Think-SFT-7B).

    `title_prefix` (e.g. "Think — ") is applied to the category title/nav
    label only, so sections stay distinguishable in the combined manifest --
    the dataset card's own `name` stays unprefixed for a cleaner display.
    """
    sample_status = json.loads(status_path.read_text()) if status_path.exists() else {}
    out_categories = []
    for src in sources:
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

        is_native = repo_id == native_dataset_repo
        # Instruct/Think-SFT sources have a card-stated `card_count` to
        # verify against; Think-DPO-7B's card gives no breakdown at all, so
        # its sources carry the live-verified `row_count` directly instead.
        count = src.get("card_count", src.get("row_count"))
        card = {
            "name": src["title"],
            "repo_id": repo_id,
            "url": f"https://huggingface.co/datasets/{repo_id}",
            "description": description,
            "metric": f"{count:,} prompts",
            # Prefer the license as stated on the Dolci card (per-source)
            # over the upstream repo's HF license tag, which is sometimes
            # missing or generic; fall back to the fetched tag.
            "license": src.get("license") or meta.get("license"),
            "downloads": None if is_native else meta.get("downloads", 0),
            "likes": None if is_native else meta.get("likes", 0),
            "gated": meta.get("gated", False),
            "sample_status": status,
            "sample_file": f"../data/dolci-samples/{safe_filename_fn(slug)}" if status == "ok" else None,
        }

        out_categories.append({
            "slug": slug,
            "title": f"{title_prefix}{src['title']}",
            "description": None,
            "url": card["url"],
            "datasets": [card],
        })
    return out_categories, sample_status


def main() -> None:
    meta_cache_path = DATA_DIR / "dolci_source_dataset_metadata.json"
    meta_cache = json.loads(meta_cache_path.read_text()) if meta_cache_path.exists() else {}

    instruct_categories, instruct_status = build_stage_categories(
        SOURCES, INSTRUCT_DATASET, instruct_safe_filename,
        DATA_DIR / "dolci_source_sample_status.json", "", meta_cache,
    )
    think_categories, think_status = build_stage_categories(
        DOLCI_THINK_SOURCES, THINK_DATASET, think_safe_filename,
        DATA_DIR / "dolci_think_source_sample_status.json", "Think — ", meta_cache,
    )
    think_dpo_categories, think_dpo_status = build_stage_categories(
        DOLCI_THINK_DPO_SOURCES, THINK_DPO_DATASET, think_dpo_safe_filename,
        DATA_DIR / "dolci_think_dpo_source_sample_status.json", "Think DPO — ", meta_cache,
    )

    meta_cache_path.write_text(json.dumps(meta_cache, indent=2))

    out_categories = instruct_categories + think_categories + think_dpo_categories
    num_sampled_ok = (
        sum(1 for v in instruct_status.values() if v == "ok")
        + sum(1 for v in think_status.values() if v == "ok")
        + sum(1 for v in think_dpo_status.values() if v == "ok")
    )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT",
        "source_model": DOLCI_INSTRUCT_MODEL,
        "num_categories": len(out_categories),
        "num_unique_datasets": len(out_categories),
        "num_sampled_ok": num_sampled_ok,
        "categories": out_categories,
    }

    out_path = DATA_DIR / "dolci-source-manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_path}: {manifest['num_categories']} categories, "
          f"{manifest['num_sampled_ok']} sampled ok")


if __name__ == "__main__":
    main()
