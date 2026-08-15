"""Build datasets/data/dolci-source-manifest.json as a 3-layer, grouped
manifest: layer 1 is one group per Dolci dataset repo covered so far (see
STAGES below), layer 2 is that dataset's native sub-source categories, and
layer 3 (16 sampled rows per category) is the existing sample-file /
sample-modal mechanism -- unchanged regardless of nesting.

Each category is filtered by that stage's identifying column -- usually
`dataset_source`/`source_dataset` (which upstream prompt collection a row
came from), but `preference_type` for Dolci-Instruct-DPO, which has no
source column at all and instead varies by preference-construction method.

This is the finer-grained sibling of build_dolci_manifest.py, which treats
a whole Dolci repo as a single (flat, 2-layer) category. To add a new
stage: write its fetch_dolci_<stage>_source_samples_duckdb.py (DATASET, a
*_SOURCES list, safe_filename()) following an existing one as a template,
then add an entry to STAGES below.
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
import fetch_dolci_think_rl_source_samples_duckdb as think_rl_mod
import fetch_dolci_instruct_rl_source_samples_duckdb as instruct_rl_mod
from fetch_dolci_source_samples import HF_TOKEN

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# One entry per Dolci stage (= one manifest group / layer-1 entry): its
# fetch module (must expose DATASET, a SOURCES-shaped list, and
# safe_filename()), the sources list attribute name (varies per module), a
# status-file name, and the group's own display metadata.
STAGES = [
    {
        "module": instruct_mod,
        "sources_attr": "SOURCES",
        "status_filename": "dolci_source_sample_status.json",
        "slug": "instruct-sft",
        "title": "Dolci-Instruct-SFT",
        "url": "https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT",
        "description": (
            "The SFT-stage mixture behind Olmo 3 Instruct's post-training pipeline "
            "(SFT &rarr; DPO &rarr; RLVR) &mdash; 2.15M examples across ~22 upstream "
            "sources, tagged directly by the card's own <code>source_dataset</code> column."
        ),
    },
    {
        "module": instruct_dpo_mod,
        "sources_attr": "DOLCI_INSTRUCT_DPO_SOURCES",
        "status_filename": "dolci_instruct_dpo_source_sample_status.json",
        "slug": "instruct-dpo",
        "title": "Dolci-Instruct-DPO",
        "url": "https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO",
        "description": (
            "260K preference pairs for Instruct 7B's DPO stage. No source-dataset column at "
            "all &mdash; split instead by <code>preference_type</code>, the 4 "
            "preference-construction methods the card describes (LLM-judged, Delta Learning, "
            "two multiturn variants)."
        ),
    },
    {
        "module": think_mod,
        "sources_attr": "DOLCI_THINK_SOURCES",
        "status_filename": "dolci_think_source_sample_status.json",
        "slug": "think-sft-7b",
        "title": "Dolci-Think-SFT-7B",
        "url": "https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B",
        "description": (
            "The reasoning-trace SFT mixture behind Olmo 3 Think 7B &mdash; largely the same "
            "upstream sources as Instruct-SFT, but with DeepSeek R1 / R1-0528 traces attached "
            "instead of stripped. 2.27M examples; <code>dataset_source</code> values are "
            "finer-grained raw identifiers than the card's 13 named categories."
        ),
    },
    {
        "module": think_dpo_mod,
        "sources_attr": "DOLCI_THINK_DPO_SOURCES",
        "status_filename": "dolci_think_dpo_source_sample_status.json",
        "slug": "think-dpo-7b",
        "title": "Dolci-Think-DPO-7B",
        "url": "https://huggingface.co/datasets/allenai/Dolci-Think-DPO-7B",
        "description": (
            "150K preference pairs for Think 7B's DPO stage, built with the "
            "<a href=\"https://arxiv.org/abs/2507.06187\" target=\"_blank\" rel=\"noopener\">Delta "
            "Learning</a> heuristic. The card gives no source breakdown at all &mdash; these 24 "
            "categories come entirely from a full <code>GROUP BY</code> on the real data."
        ),
    },
    {
        "module": think_rl_mod,
        "sources_attr": "DOLCI_THINK_RL_SOURCES",
        "status_filename": "dolci_think_rl_source_sample_status.json",
        "slug": "think-rl-7b",
        "title": "Dolci-Think-RL-7B",
        "url": "https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B",
        "description": (
            "102K RLVR prompts for Think 7B's RL stage, spanning math, code, precise "
            "instruction-following, and general chat."
        ),
    },
    {
        "module": instruct_rl_mod,
        "sources_attr": "DOLCI_INSTRUCT_RL_SOURCES",
        "status_filename": "dolci_instruct_rl_source_sample_status.json",
        "slug": "instruct-rl",
        "title": "Dolci-Instruct-RL",
        "url": "https://huggingface.co/datasets/allenai/Dolci-Instruct-RL",
        "description": (
            "170K RLVR prompts for Instruct 7B's RL stage, the same four domains as "
            "Think-RL-7B."
        ),
    },
]

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


def build_stage_categories(sources, native_dataset_repo, safe_filename_fn, status_path, meta_cache):
    """Build layer-2 categories (one per native sub-source) for one Dolci stage."""
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
        # (Think-DPO-7B, Instruct-DPO, the RL datasets) carry the
        # live-verified `row_count` directly instead.
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
            "title": src["title"],
            "description": None,
            "url": card["url"],
            "datasets": [card],
            "count": count,
        })
    return out_categories, sample_status


def main() -> None:
    meta_cache_path = DATA_DIR / "dolci_source_dataset_metadata.json"
    meta_cache = json.loads(meta_cache_path.read_text()) if meta_cache_path.exists() else {}

    groups = []
    total_categories = 0
    num_sampled_ok = 0
    for stage in STAGES:
        categories, status = build_stage_categories(
            getattr(stage["module"], stage["sources_attr"]), stage["module"].DATASET,
            stage["module"].safe_filename, DATA_DIR / stage["status_filename"], meta_cache,
        )
        total_prompts = sum(cat["count"] for cat in categories)
        for cat in categories:
            del cat["count"]

        groups.append({
            "slug": stage["slug"],
            "title": stage["title"],
            "description": stage["description"],
            "url": stage["url"],
            "metric": f"{total_prompts:,} prompts",
            "categories": categories,
        })
        total_categories += len(categories)
        num_sampled_ok += sum(1 for v in status.values() if v == "ok")

    meta_cache_path.write_text(json.dumps(meta_cache, indent=2))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": "https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT",
        "source_model": DOLCI_INSTRUCT_MODEL,
        "num_groups": len(groups),
        "num_categories": total_categories,
        "num_unique_datasets": total_categories,
        "num_sampled_ok": num_sampled_ok,
        "groups": groups,
    }

    out_path = DATA_DIR / "dolci-source-manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_path}: {manifest['num_groups']} groups, {manifest['num_categories']} categories, "
          f"{manifest['num_sampled_ok']} sampled ok")


if __name__ == "__main__":
    main()
