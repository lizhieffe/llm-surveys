"""Fetch and group task cards for Agents' Last Exam (ALE, arXiv:2606.05405,
UC Berkeley RDI): "https://github.com/rdi-berkeley/agents-last-exam".

ALE is an agent evaluation benchmark, not a text/vision QA dataset like the
other sources here -- each "example" is a full long-horizon task (a written
brief, required software, input files, and a hidden grading reference) meant
to be run by an agent in a real OS sandbox, not answered from a row of text.
Only the **task cards** (structured metadata: title, prompt, taxonomy, input
file descriptors -- no images, no hidden reference/grader) are publicly
browsable; ALE ships them as their own clean, single-parquet HF dataset,
`agents-last-exam/agents-last-exam` (one of three companion repos -- the
input-data repo mixes incompatible formats per split with no working viewer,
and the reference/grader repo is gated -- so this is deliberately the only
one of the three used here).

Unlike every other source in this survey, ALE needs no per-category HTTP
fetch: the whole public release is 153 rows in one parquet, small enough to
pull in two /rows calls and group locally by its own `taxonomy` field --
domain_code (14 present in the public sample, though the paper's full,
mostly-private taxonomy claims 13 "industry clusters") becomes the
category, and subdomain_code (51 present, of a claimed 55) becomes each
"dataset" card, holding that subdomain's tasks as its sampled rows. No
subdomain in the public sample exceeds NUM_EXAMPLES tasks, so nothing is
actually subsampled -- every task in a subdomain is shown.

`domain_code` has no human-readable name in the data (only `subdomain_name`
does); DOMAIN_TITLES below are display names this script authors for
readability, not something ALE itself provides.
"""

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "ale-samples"
REPO_ID = "agents-last-exam/agents-last-exam"
NUM_EXAMPLES = 16
TRUNCATE_CHARS = 4000
SERVER = "https://datasets-server.huggingface.co"

DOMAIN_TITLES = {
    "agriculture_env": "Agriculture & Environment",
    "business_finance": "Business & Finance",
    "computing_math": "Computing & Math",
    "education_info": "Education & Information",
    "engineering": "Engineering",
    "health_medicine": "Health & Medicine",
    "legal": "Legal",
    "life_sciences": "Life Sciences",
    "other": "Other",
    "physical_sciences": "Physical Sciences",
    "psychology_neuro": "Psychology & Neuroscience",
    "social_sciences": "Social Sciences",
    "transport_safety": "Transport & Safety",
    "visual_media": "Visual Media & Design",
}


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-surveys"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def truncate(value):
    if isinstance(value, str) and len(value) > TRUNCATE_CHARS:
        return value[:TRUNCATE_CHARS] + f"... [truncated, {len(value):,} chars total]"
    if isinstance(value, list):
        return [truncate(v) for v in value]
    if isinstance(value, dict):
        return {k: truncate(v) for k, v in value.items()}
    return value


def subdomain_slug(domain_code: str, subdomain_code: str) -> str:
    return f"{domain_code}__{subdomain_code}"


def fetch_all_rows() -> list[dict]:
    size_resp = _get(f"{SERVER}/size?dataset={REPO_ID}")
    num_rows = size_resp["size"]["dataset"]["num_rows"]

    splits_resp = _get(f"{SERVER}/splits?dataset={REPO_ID}")
    config, split = splits_resp["splits"][0]["config"], splits_resp["splits"][0]["split"]

    rows, features, offset = [], None, 0
    while offset < num_rows:
        resp = _get(f"{SERVER}/rows?dataset={REPO_ID}&config={config}&split={split}&offset={offset}&length=100")
        if features is None:
            features = resp.get("features", [])
        rows.extend(r["row"] for r in resp["rows"])
        offset += 100

    return config, split, features, rows


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"fetching all rows from {REPO_ID} ...")
    config, split, features, rows = fetch_all_rows()
    print(f"got {len(rows)} rows")

    by_domain: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    subdomain_names: dict[str, str] = {}
    for row in rows:
        tax = row["taxonomy"]
        by_domain[tax["domain_code"]][tax["subdomain_code"]].append(row)
        subdomain_names[tax["subdomain_code"]] = tax["subdomain_name"]

    status = {}
    domains_out = []
    for domain_code in sorted(by_domain):
        subdomains_out = []
        for subdomain_code in sorted(by_domain[domain_code]):
            subdomain_rows = by_domain[domain_code][subdomain_code]
            slug = subdomain_slug(domain_code, subdomain_code)
            sample = {
                "status": "ok",
                "config": config,
                "split": split,
                "features": features,
                "rows": [truncate(r) for r in subdomain_rows[:NUM_EXAMPLES]],
            }
            if len(subdomain_rows) > NUM_EXAMPLES:
                sample["num_rows_total"] = len(subdomain_rows)
            (SAMPLES_DIR / f"{slug}.json").write_text(json.dumps(sample, indent=2, default=str))
            status[slug] = {"status": "ok", "count": len(subdomain_rows)}
            subdomains_out.append({
                "subdomain_code": subdomain_code,
                "subdomain_name": subdomain_names[subdomain_code],
                "count": len(subdomain_rows),
                "slug": slug,
            })
        domains_out.append({"domain_code": domain_code, "subdomains": subdomains_out})

    (DATA_DIR / "ale_sample_status.json").write_text(json.dumps(status, indent=2))
    (DATA_DIR / "ale_domains.json").write_text(json.dumps(domains_out, indent=2))

    print(f"\n{len(by_domain)} domains, {sum(len(v) for v in by_domain.values())} subdomains, {len(rows)} tasks total")


if __name__ == "__main__":
    main()
