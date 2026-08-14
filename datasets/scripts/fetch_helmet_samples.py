"""Fetch 16 randomly-sampled rows for each HELMET evaluation dataset.

HELMET (arXiv:2410.02694, Princeton NLP) ships its preprocessed eval data as a
single 34GB tarball (princeton-nlp/HELMET on the Hub has no per-task viewer).
User "xiaoyuanliu" separately re-uploaded the same preprocessed eval files as
individual per-task Hugging Face dataset repos (one config/split each, viewer
enabled), which is what this script reads. The category/task list and mapping
to those repos was verified by hand against Table 3 of the paper and by
inspecting each repo's fields (e.g. RULER MK Needle vs MK UUID were
disambiguated via the `type_needle_v` field).

HELMET rows contain full long-context prompts (up to ~128K tokens), so any
string field longer than TRUNCATE_CHARS is truncated when saving -- otherwise
a 16-row sample file would be tens of MB and pointless to render inline.
"""

import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "helmet-samples"
NUM_EXAMPLES = 16
TRUNCATE_CHARS = 4000
TRUNCATE_LIST_ITEMS = 5
SERVER = "https://datasets-server.huggingface.co"
SEED = 42

# Table 3 of the HELMET paper (arXiv:2410.02694), with each dataset mapped to
# the xiaoyuanliu/HELMET_* mirror of the official preprocessed eval file.
CATEGORIES = [
    {
        "title": "Retrieval-augmented generation",
        "description": "Answer a question given retrieved documents mixed with distractors.",
        "datasets": [
            {"name": "Natural Questions", "metric": "SubEM", "description": "Factoid question answering",
             "repo_id": "xiaoyuanliu/HELMET_kilt_nq_nq-dev-multikilt_1000_k1000_dep6_eval"},
            {"name": "TriviaQA", "metric": "SubEM", "description": "Trivia question answering",
             "repo_id": "xiaoyuanliu/HELMET_kilt_triviaqa_triviaqa-dev-multikilt_1000_k1000_dep6_eval"},
            {"name": "PopQA", "metric": "SubEM", "description": "Long-tail entity question answering",
             "repo_id": "xiaoyuanliu/HELMET_kilt_popqa_3_popqa_test_1000_k1000_dep6_eval"},
            {"name": "HotpotQA", "metric": "SubEM", "description": "Multi-hop question answering",
             "repo_id": "xiaoyuanliu/HELMET_kilt_hotpotqa_hotpotqa-dev-multikilt_1000_k1000_dep3_eval"},
        ],
    },
    {
        "title": "Generation with citations",
        "description": "Answer a question and cite which retrieved passages support each claim.",
        "datasets": [
            {"name": "ALCE ASQA", "metric": "Recall, Citation", "description": "Answer ambiguous questions with citations",
             "repo_id": "xiaoyuanliu/HELMET_alce_asqa_700_asqa_eval_gtr_top2000_eval"},
            {"name": "ALCE QAMPARI", "metric": "Recall, Citation", "description": "Answer factoid questions with citations",
             "repo_id": "xiaoyuanliu/HELMET_alce_qampari_700_qampari_eval_gtr_top2000_eval"},
        ],
    },
    {
        "title": "Passage re-ranking",
        "description": "Re-rank a shuffled list of retrieved passages by relevance to a query.",
        "datasets": [
            {"name": "MS MARCO", "metric": "NDCG@10", "description": "Rerank passages for a query",
             "repo_id": "xiaoyuanliu/HELMET_msmarco_rerank_psg_test_reranking_data_k1000_dep3_eval"},
        ],
    },
    {
        "title": "Many-shot in-context learning",
        "description": "Classify a query given hundreds to thousands of labeled in-context examples.",
        "datasets": [
            {"name": "TREC Coarse", "metric": "Accuracy", "description": "Question type classification, 6 labels",
             "repo_id": "xiaoyuanliu/HELMET_icl_trec_coarse_6600shot_balance__eval"},
            {"name": "TREC Fine", "metric": "Accuracy", "description": "Question type classification, 50 labels",
             "repo_id": "xiaoyuanliu/HELMET_icl_trec_fine_6400shot_balance__eval"},
            {"name": "NLU", "metric": "Accuracy", "description": "Task intent classification, 68 labels",
             "repo_id": "xiaoyuanliu/HELMET_icl_nlu_8296shot_balance__eval"},
            {"name": "BANKING77", "metric": "Accuracy", "description": "Banking intent classification, 77 labels",
             "repo_id": "xiaoyuanliu/HELMET_icl_banking77_5900shot_balance__eval"},
            {"name": "CLINC150", "metric": "Accuracy", "description": "Intent classification, 151 labels",
             "repo_id": "xiaoyuanliu/HELMET_icl_clinic150_7050shot_balance__eval"},
        ],
    },
    {
        "title": "Long-document QA",
        "description": "Answer a question given a full-length book, movie script, or novel.",
        "datasets": [
            {"name": "NarrativeQA", "metric": "Model-based", "description": "Book and movie script QA",
             "repo_id": "xiaoyuanliu/HELMET_narrativeqa_130772__eval"},
            {"name": "∞BENCH QA", "metric": "ROUGE F1", "description": "Novel QA with entity replacement",
             "repo_id": "xiaoyuanliu/HELMET_infbench_qa_eng_130862__eval"},
            {"name": "∞BENCH MC", "metric": "Accuracy", "description": "Novel multiple-choice QA with entity replacement",
             "repo_id": "xiaoyuanliu/HELMET_infbench_choice_eng_130862__eval"},
        ],
    },
    {
        "title": "Summarization",
        "description": "Summarize one or more long documents.",
        "datasets": [
            {"name": "∞BENCH Sum", "metric": "Model-based", "description": "Novel summarization with entity replacement",
             "repo_id": "xiaoyuanliu/HELMET_infbench_sum_eng_129672__eval"},
            {"name": "Multi-LexSum", "metric": "Model-based", "description": "Summarizing multiple legal documents",
             "repo_id": "xiaoyuanliu/HELMET_multi_lexsum_130372__eval"},
        ],
    },
    {
        "title": "Synthetic recall",
        "description": "Retrieve a specific value planted among many distractors in a long, noisy context.",
        "datasets": [
            {"name": "JSON KV", "metric": "SubEM", "description": "Retrieve a key in a JSON dictionary",
             "repo_id": "xiaoyuanliu/HELMET_json_kv_test_k1800_dep6_eval"},
            {"name": "RULER MK Needle", "metric": "SubEM", "description": "Retrieve the needle (a number) among noisy needles",
             "repo_id": "xiaoyuanliu/HELMET_ruler_niah_mk_2_validation_131072_eval"},
            {"name": "RULER MK UUID", "metric": "SubEM", "description": "Retrieve the needle (a UUID) among noisy needles",
             "repo_id": "xiaoyuanliu/HELMET_ruler_niah_mk_3_validation_131072_eval"},
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


def truncate(value):
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


def fetch_sample(repo_id: str, rng: random.Random) -> dict:
    try:
        size_resp = _get(f"{SERVER}/size?dataset={repo_id}")
        splits_resp = _get(f"{SERVER}/splits?dataset={repo_id}")
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)[:300]}

    splits = splits_resp.get("splits", [])
    if not splits:
        return {"status": "no_viewer", "error": "no splits returned"}
    config, split = splits[0]["config"], splits[0]["split"]
    num_rows = size_resp["size"]["dataset"]["num_rows"]

    n = min(NUM_EXAMPLES, num_rows)
    indices = sorted(rng.sample(range(num_rows), n))

    rows, features = [], None
    for idx in indices:
        try:
            resp = _get(f"{SERVER}/rows?dataset={repo_id}&config={config}&split={split}&offset={idx}&length=1")
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "error": str(e)[:300]}
        if features is None:
            features = resp.get("features", [])
        for r in resp.get("rows", []):
            rows.append(truncate(r["row"]))
        time.sleep(0.3)

    return {"status": "ok", "config": config, "split": split, "features": features, "rows": rows,
            "num_rows_total": num_rows}


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    status_path = DATA_DIR / "helmet_sample_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    all_repos = [d["repo_id"] for cat in CATEGORIES for d in cat["datasets"]]
    for i, repo_id in enumerate(all_repos, 1):
        out_path = SAMPLES_DIR / safe_filename(repo_id)
        if out_path.exists() and status.get(repo_id) == "ok":
            print(f"[{i}/{len(all_repos)}] skip (cached) {repo_id}")
            continue

        print(f"[{i}/{len(all_repos)}] fetching {repo_id} ...", end=" ", flush=True)
        rng = random.Random(SEED)  # fixed seed per dataset for reproducible sampling
        result = fetch_sample(repo_id, rng)
        status[repo_id] = result["status"]
        print(result["status"])
        if result["status"] == "ok":
            out_path.write_text(json.dumps(result, indent=2, default=str))
        else:
            print(f"    -> {result.get('error', '')}")
        status_path.write_text(json.dumps(status, indent=2))
        time.sleep(0.5)

    ok = sum(1 for v in status.values() if v == "ok")
    print(f"\nDone. {ok}/{len(all_repos)} datasets sampled successfully.")


if __name__ == "__main__":
    main()
