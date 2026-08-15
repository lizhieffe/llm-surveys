"""Build datasets/data/dolci-source-manifest.json: one category per native
sub-set, across all Dolci stages covered so far (see STAGES below), each
carrying its own 16 sampled rows filtered by that stage's identifying
column -- usually `dataset_source`/`source_dataset` (which upstream prompt
collection a row came from), but `preference_type` for Dolci-Instruct-DPO,
which has no source column at all and instead varies by
preference-construction method.

This is the finer-grained sibling of build_dolci_manifest.py, which treats
a whole Dolci repo as a single category. To add a new stage: write its
fetch_dolci_<stage>_source_samples_duckdb.py (DATASET, a *_SOURCES list,
safe_filename()) following an existing one as a template, then add a row
to STAGES below.
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import fetch_dolci_source_samples as instruct_mod
import fetch_dolci_instruct_dpo_source_samples_duckdb as instruct_dpo_mod
import fetch_dolci_think_source_samples_duckdb as think_mod
import fetch_dolci_think_dpo_source_samples_duckdb as think_dpo_mod
from fetch_dolci_source_samples import HF_TOKEN

# Each Dolci stage covered so far: its fetch module (must expose DATASET,
# a SOURCES-shaped list, and safe_filename()), the sources list attribute
# name (varies per module), a status-file name, and a category title
# prefix ("" for the first/primary stage, distinguishing prefixes after).
STAGES = [
    (instruct_mod, "SOURCES", "dolci_source_sample_status.json", ""),
    (think_mod, "DOLCI_THINK_SOURCES", "dolci_think_source_sample_status.json", "Think — "),
    (think_dpo_mod, "DOLCI_THINK_DPO_SOURCES", "dolci_think_dpo_source_sample_status.json", "Think DPO — "),
    (instruct_dpo_mod, "DOLCI_INSTRUCT_DPO_SOURCES", "dolci_instruct_dpo_source_sample_status.json", "Instruct DPO — "),
]

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
    """Build manifest categories for one Dolci stage.

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
        # Instruct-SFT/Think-SFT sources have a card-stated `card_count` to
        # verify against; stages whose card gives no per-source breakdown
        # (Think-DPO-7B, Instruct-DPO) carry the live-verified `row_count`
        # directly instead.
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

    out_categories = []
    num_sampled_ok = 0
    for module, sources_attr, status_filename, title_prefix in STAGES:
        categories, status = build_stage_categories(
            getattr(module, sources_attr), module.DATASET, module.safe_filename,
            DATA_DIR / status_filename, title_prefix, meta_cache,
        )
        out_categories += categories
        num_sampled_ok += sum(1 for v in status.values() if v == "ok")

    meta_cache_path.write_text(json.dumps(meta_cache, indent=2))

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
