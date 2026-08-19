"""Build datasets/data/ale-manifest.json from fetch_ale_samples.py's output
(ale_domains.json, ale_sample_status.json) plus the shared HF repo metadata.
"""

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fetch_ale_samples import DOMAIN_TITLES, REPO_ID

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALE_GITHUB = "https://github.com/rdi-berkeley/agents-last-exam"
ALE_PAPER = "https://arxiv.org/abs/2606.05405"
ALE_HF_URL = f"https://huggingface.co/datasets/{REPO_ID}"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-surveys"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_repo_metadata() -> dict:
    try:
        data = _get(f"https://huggingface.co/api/datasets/{REPO_ID}")
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
    domains = json.loads((DATA_DIR / "ale_domains.json").read_text())
    meta = fetch_repo_metadata()

    total_tasks = sum(sd["count"] for d in domains for sd in d["subdomains"])

    out_categories = []
    for d in domains:
        domain_code = d["domain_code"]
        title = DOMAIN_TITLES.get(domain_code, domain_code.replace("_", " ").title())
        domain_task_count = sum(sd["count"] for sd in d["subdomains"])

        datasets = []
        for sd in d["subdomains"]:
            datasets.append({
                "name": sd["subdomain_name"],
                "metric": f"{sd['count']} task{'s' if sd['count'] != 1 else ''}",
                "repo_id": REPO_ID,
                "url": ALE_HF_URL,
                "description": f"O*NET/SOC subdomain “{sd['subdomain_code']}” within the "
                                f"{title} cluster — {sd['count']} ALE task card"
                                f"{'s' if sd['count'] != 1 else ''}.",
                "downloads": meta.get("downloads", 0),
                "likes": meta.get("likes", 0),
                "license": meta.get("license"),
                "gated": meta.get("gated", False),
                "sample_status": "ok",
                "sample_file": f"../data/ale-samples/{sd['slug']}.json",
            })

        out_categories.append({
            "slug": domain_code,
            "title": title,
            "description": f"{domain_task_count} task card{'s' if domain_task_count != 1 else ''} "
                            f"across {len(d['subdomains'])} O*NET/SOC subdomain"
                            f"{'s' if len(d['subdomains']) != 1 else ''}.",
            "url": ALE_HF_URL,
            "datasets": datasets,
        })

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_url": ALE_PAPER,
        "source_github": ALE_GITHUB,
        "source_official_hf": ALE_HF_URL,
        "num_categories": len(out_categories),
        "num_unique_datasets": sum(len(c["datasets"]) for c in out_categories),
        "num_sampled_ok": sum(len(c["datasets"]) for c in out_categories),
        "num_tasks_total": total_tasks,
        "categories": out_categories,
    }

    out_path = DATA_DIR / "ale-manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_path}: {manifest['num_categories']} categories (domains), "
          f"{manifest['num_unique_datasets']} subdomain cards, {total_tasks} tasks total")


if __name__ == "__main__":
    main()
