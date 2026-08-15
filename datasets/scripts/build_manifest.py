"""Combine categories.json + dataset_metadata.json + sample_status.json into a
single data/manifest.json for the frontend to consume.

Full 16-row samples stay in their own per-dataset files under data/samples/ and
are fetched lazily by the page, so the manifest itself stays small.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Nemotron's own HF collections mix pre-training and post-training data under
# one brand (see https://developer.nvidia.com/topics/ai/nemotron and the NVIDIA
# collection descriptions in categories.json), so each category is tagged with
# a training-stage badge where there's clear textual evidence for one -- never
# guessed. `classify_stage()` looks for NVIDIA's own explicit wording
# ("pre-training", "SFT", "reward model", "RLHF", "reinforcement learning",
# "instruction-following", ...) in that category's own collection description;
# categories with both a pretraining signal and a post-training signal are
# "Mixed". Two categories get a manual override because the evidence lives one
# level up rather than in their own description text: Code & SWE explicitly
# groups "code pretraining" alongside competitive-programming/SWE tasks the
# same way its sibling "Math & Reasoning" category does (which does state
# "Covers SFT, RL, and pretraining data" directly), and Nemotron 4 340B's
# description says it "Includes Base, Instruct, and Reward models" -- a
# pretrained Base model plus post-trained Instruct/Reward models. Everything
# else is left unbadged rather than assigned a low-confidence guess.
_STAGE_OVERRIDES = {
    "Nemotron Code & SWE": "Mixed",
    "Nemotron 4 340B": "Mixed",
}
_PRETRAIN_SIGNALS = ("pre-training", "pretraining", "pretrain")
_POSTTRAIN_SIGNALS = (
    "post-training", "post training", "sft", "supervised fine-tun",
    "reward model", "rlhf", "reinforcement learning", "cascade rl",
    "preference", "instruction-following", "instruction following",
)


def classify_stage(title: str, description: str) -> str | None:
    if title in _STAGE_OVERRIDES:
        return _STAGE_OVERRIDES[title]
    text = f"{title} {description}".lower()
    has_pretrain = any(s in text for s in _PRETRAIN_SIGNALS)
    has_posttrain = any(s in text for s in _POSTTRAIN_SIGNALS)
    if has_pretrain and has_posttrain:
        return "Mixed"
    if has_pretrain:
        return "Pre-training"
    if has_posttrain:
        return "Post-training"
    return None


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
                    "sample_file": f"../data/samples/{repo_id.replace('/', '__')}.json" if status == "ok" else None,
                }
            )
        out_categories.append(
            {
                "slug": cat["slug"],
                "title": cat["title"],
                "description": cat["description"],
                "url": cat["url"],
                "stage": classify_stage(cat["title"], cat["description"]),
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
