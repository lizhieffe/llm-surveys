"""Fetch 16 randomly-sampled rows for each MMLongBench task dataset.

MMLongBench (arXiv:2505.10610, NeurIPS 2025) is a long-context VISION-language
benchmark: 13,331 examples across 5 categories (Table 2 of the paper), 16
named datasets. Unlike HELMET, its official repo (ZhaoweiWang/MMLongBench on
the Hub) ships as per-category tarballs with images referenced by relative
path (not embedded) and a broken auto-generated viewer (schema differs across
its five context-length variants). There is no single clean per-task mirror
the way xiaoyuanliu's HELMET re-upload was, so each of the 16 tasks here is
instead mapped by hand to the best available public source:

- Where MMLongBench reuses a well-known upstream benchmark that has its own
  clean, viewer-enabled Hub repo with real embedded images (Stanford Cars,
  Food101, SUN397, GovReport, Multi-LexSum, and -- via community re-uploads
  with embedded images by user "shrekwang" -- MMLongBench-Doc and
  LongDocURL), that repo is used directly.
- iNaturalist2021's only easily-viewable mirror is a 12K-image subset
  (demoyolo/inaturalist-12k), not MMLongBench's exact 50-species sample, but
  a reasonable stand-in for "what does this task's data look like".
- Visual Haystack and MM-NIAH's public mirrors (tsunghanwu/visual_haystacks,
  OpenGVLab/MM-NIAH) only exposed ONE blended task variant each via the
  viewer, not the paper's finer single/multi or ret/count/reason split --
  sampling many offsets across each dataset confirmed this (see git history
  / commit message for the investigation). They're listed here as one merged
  entry per source rather than split into fake-looking sub-cards.
- InfoSeek and SlideVQA have no working public dataset-viewer source found
  (schema errors / gated / persistent server errors across every candidate
  repo and split tried) and are marked unavailable, same as any other
  gated/broken dataset elsewhere on this site.
- ViQuAE, Visual Haystack, and MM-NIAH reference images by filename/relative
  path into external corpora (Wikipedia Commons, COCO, OBELICS) rather than
  embedding them, so their samples are text-only (no inline thumbnails).

Rows may contain embedded images (HF Image feature -> {"src", "height",
"width"} dicts); these are preserved as-is (not truncated) so the frontend
can render them. Long strings are still truncated, and long lists are capped,
same as the HELMET script.
"""

import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "mmlongbench-samples"
NUM_EXAMPLES = 16
TRUNCATE_CHARS = 4000
TRUNCATE_LIST_ITEMS = 8
SERVER = "https://datasets-server.huggingface.co"
SEED = 42

# Table 2 of the MMLongBench paper (arXiv:2505.10610), mapped to public HF
# repos. `split`, when set, overrides the "first available split" default.
CATEGORIES = [
    {
        "title": "Visual RAG",
        "description": "Answer a factoid question about a named entity given a long, distractor-filled context of retrieved passages.",
        "datasets": [
            {"name": "InfoSeek", "metric": "SubEM", "description": "Long-tail entity question answering",
             "repo_id": "Lk123/InfoSeek", "split": "train"},
            {"name": "ViQuAE", "metric": "SubEM", "description": "Question answering based on TriviaQA",
             "repo_id": "PaulLerner/viquae_dataset", "split": "test"},
        ],
    },
    {
        "title": "Needle-in-a-Haystack",
        "description": "Retrieve, count, or reason about a small planted target among many distractor images/text in a long context.",
        "datasets": [
            {"name": "Visual Haystack (VH)", "metric": "Acc", "description": "Retrieve image(s) from a large photo album given a text query (merges the paper's VH-Single/VH-Multi split -- the public mirror only exposes one blended variant)",
             "repo_id": "tsunghanwu/visual_haystacks", "split": "train"},
            {"name": "MM-NIAH", "metric": "SubEM/Acc", "description": "Retrieve, count, or reason about text/image needles planted in long interleaved web documents (merges the paper's Ret/Count/Reason split -- the public mirror only exposes one blended task variant)",
             "repo_id": "OpenGVLab/MM-NIAH", "config": "val", "split": "val"},
        ],
    },
    {
        "title": "Many-Shot In-Context Learning",
        "description": "Classify an image given up to hundreds of in-context labeled exemplars (labels mapped to opaque IDs, not natural-language class names).",
        "datasets": [
            {"name": "Stanford Cars", "metric": "Acc", "description": "50-category car classification",
             "repo_id": "tanganke/stanford_cars", "split": "test"},
            {"name": "Food101", "metric": "Acc", "description": "50-category food classification",
             "repo_id": "ethz/food101", "split": "validation"},
            {"name": "SUN397", "metric": "Acc", "description": "50-category scene classification",
             "repo_id": "tanganke/sun397", "split": "test"},
            {"name": "iNat2021", "metric": "Acc", "description": "50-category species classification",
             "repo_id": "demoyolo/inaturalist-12k", "split": "validation"},
        ],
    },
    {
        "title": "Summarization",
        "description": "Summarize one or more long, image-based PDF documents.",
        "datasets": [
            {"name": "GovReport", "metric": "Model-based", "description": "Summarizing government reports in PDF",
             "repo_id": "ccdv/govreport-summarization", "config": "document", "split": "test"},
            {"name": "Multi-LexSum", "metric": "Model-based", "description": "Summarizing multiple legal documents in PDF",
             "repo_id": "xiaoyuanliu/HELMET_multi_lexsum_130372__eval", "split": "train"},
        ],
    },
    {
        "title": "Long-Document VQA",
        "description": "Answer a question requiring reasoning over multiple images/pages within a long PDF document or slide deck.",
        "datasets": [
            {"name": "MMLongBench-Doc", "metric": "SubEM/Acc", "description": "Long PDF document VQA",
             "repo_id": "shrekwang/bam_mmlongbench-doc", "split": "train"},
            {"name": "LongDocURL", "metric": "SubEM/Acc", "description": "Long PDF document VQA",
             "repo_id": "shrekwang/bam_longdocurl", "split": "train"},
            {"name": "SlideVQA", "metric": "SubEM/Acc", "description": "Slide deck understanding and reasoning",
             "repo_id": "Ahren09/SlideVQA", "split": "test"},
        ],
    },
]


def _get(url: str, timeout: int = 60, max_retries: int = 5) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-surveys"})
    backoff = 3.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise


def is_image_cell(value) -> bool:
    return isinstance(value, dict) and "src" in value and ("height" in value or "width" in value)


def truncate(value):
    if is_image_cell(value):
        return value  # keep image cells intact for the frontend to render
    if isinstance(value, str) and len(value) > TRUNCATE_CHARS:
        return value[:TRUNCATE_CHARS] + f"... [truncated, {len(value):,} chars total]"
    if isinstance(value, list):
        kept = [truncate(v) for v in value[:TRUNCATE_LIST_ITEMS]]
        if len(value) > TRUNCATE_LIST_ITEMS:
            kept.append(f"... [{len(value) - TRUNCATE_LIST_ITEMS} more items omitted, {len(value)} total]")
        return kept
    if isinstance(value, dict):
        return {k: truncate(v) for k, v in value.items()}
    return value


def safe_filename(repo_id: str) -> str:
    return repo_id.replace("/", "__") + ".json"


def fetch_sample(ds: dict, rng: random.Random) -> dict:
    repo_id = ds["repo_id"]
    try:
        size_resp = _get(f"{SERVER}/size?dataset={repo_id}")
        splits_resp = _get(f"{SERVER}/splits?dataset={repo_id}")
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)[:300]}

    if "error" in size_resp or "error" in splits_resp:
        return {"status": "no_viewer", "error": (size_resp.get("error") or splits_resp.get("error", ""))[:300]}

    splits = splits_resp.get("splits", [])
    if not splits:
        return {"status": "no_viewer", "error": "no splits returned"}

    chosen = splits[0]
    for s in splits:
        if s["split"] == ds.get("split") and (ds.get("config") is None or s["config"] == ds.get("config")):
            chosen = s
            break
    config, split = chosen["config"], chosen["split"]

    try:
        num_rows = next(
            s["num_rows"] for s in size_resp["size"]["splits"]
            if s["config"] == config and s["split"] == split
        )
    except (KeyError, StopIteration):
        return {"status": "error", "error": "could not determine row count"}

    n = min(NUM_EXAMPLES, num_rows)
    indices = sorted(rng.sample(range(num_rows), n))

    rows, features = [], None
    for idx in indices:
        try:
            resp = _get(f"{SERVER}/rows?dataset={repo_id}&config={config}&split={split}&offset={idx}&length=1")
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": str(e)[:300]}
        if "error" in resp:
            return {"status": "no_viewer", "error": resp["error"][:300]}
        if features is None:
            features = resp.get("features", [])
        for r in resp.get("rows", []):
            rows.append(truncate(r["row"]))
        time.sleep(0.3)

    return {"status": "ok", "config": config, "split": split, "features": features, "rows": rows,
            "num_rows_total": num_rows}


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "mmlongbench_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    all_datasets = [d for cat in CATEGORIES for d in cat["datasets"]]
    for i, ds in enumerate(all_datasets, 1):
        repo_id = ds["repo_id"]
        out_path = SAMPLES_DIR / safe_filename(repo_id)
        if out_path.exists() and status.get(repo_id) == "ok":
            print(f"[{i}/{len(all_datasets)}] skip (cached) {repo_id}")
            continue

        print(f"[{i}/{len(all_datasets)}] fetching {repo_id} ...", end=" ", flush=True)
        rng = random.Random(SEED)
        result = fetch_sample(ds, rng)
        status[repo_id] = result["status"]
        print(result["status"])
        if result["status"] == "ok":
            out_path.write_text(json.dumps(result, indent=2, default=str))
        else:
            print(f"    -> {result.get('error', '')}")
        status_path.write_text(json.dumps(status, indent=2))
        time.sleep(0.5)

    ok = sum(1 for v in status.values() if v == "ok")
    print(f"\nDone. {ok}/{len(all_datasets)} datasets sampled successfully.")


if __name__ == "__main__":
    main()
