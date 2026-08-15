# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Source for [lizhieffe.github.io/llm-surveys](https://lizhieffe.github.io/llm-surveys/) — a static,
dependency-free site with two survey tracks, each figure linked back to its primary source:

- **Dataset study** (`datasets/`) — per-source pages surveying the datasets behind published LLMs
  (Nemotron, HELMET, MMLongBench, Dolci), each with 16 sampled rows per dataset pulled from the
  Hugging Face dataset viewer / parquet exports.
- **Training detail study** (`training/`) — a hand-curated table of training hardware, throughput,
  token counts, and estimated GPU-hours, model by model.

No build step, no package manager, no test suite: plain HTML/CSS/JS served directly by GitHub Pages,
generated/refreshed by standalone Python (stdlib-only, except where noted) scripts that write static
JSON the frontend fetches at runtime.

## Commands

There's nothing to build, lint, or test. The only "commands" are the per-source regeneration
pipelines below (run from `datasets/`), and previewing locally with `python3 -m http.server`.

## Dataset study architecture (`datasets/`)

`datasets/index.html` is a chooser page linking to one full page per source
(`datasets/nemotron/`, `datasets/helmet/`, `datasets/mmlongbench/`, `datasets/dolci/`) — each source
gets its own URL rather than being stacked on one page, since Nemotron alone has 100+ dataset cards.

Rendering is shared via `datasets/shared.js` (`DatasetSurvey.init(config)`): category nav, dataset
cards, and the sample-row modal are all common code. Each subpage's inline `<script>` only supplies
`config.manifestUrl` (that source's manifest JSON) and intro copy. Manifest shape:

```
{categories: [{title, description, url, datasets: [{repo_id, url, description, license,
  downloads, likes, sample_status, sample_file, ...}]}]}
```

`sample_file` paths are relative to the subpage (e.g. `../data/samples/x.json`), fetched lazily on
"View N examples" click so initial page load stays light. Long text fields and long lists are
truncated when samples are written (not at render time), so sample files stay small.

To add a new source: write a script producing a manifest in the shape above, add
`datasets/<name>/index.html` (copy an existing subpage, change the `init()` config), and add a card
to `datasets/index.html`.

### Per-source pipelines

Each source is a **fetch step** (queries HF for raw samples, caches per-dataset JSON under
`datasets/data/<source>-samples/` or `datasets/data/samples/`, keyed so re-running only retries
failures) followed by a **build step** (`build_*_manifest.py`, combines fetch output + metadata into
the single manifest JSON the frontend loads).

- **Nemotron** (`fetch_collections.py` → `fetch_dataset_metadata.py` → `sample_datasets.py` →
  `build_manifest.py`): fully automated from the HF collections API — no hand-curated mapping.
  Uses the `datasets-server` rows API (first 16 rows), which works even for multi-TB datasets since
  it reads a pre-built server-side parquet export. Gated/viewer-less datasets are recorded as such,
  not silently skipped. A dataset can legitimately appear in more than one category (mirrors
  overlapping NVIDIA collections).

- **HELMET** (`fetch_helmet_samples.py` → `build_helmet_manifest.py`): category→dataset→repo mapping
  is hand-curated in `CATEGORIES` (from the paper's Table 3), because the official repo ships one
  34GB tarball with no per-task viewer — samples instead come from a verified community re-upload.
  True random sampling (fixed seed 42, via `/size` + per-index row fetches), with a 4,000-char
  truncation per string field so sample files don't balloon on long-context rows.

- **MMLongBench** (`fetch_mmlongbench_samples.py` → `build_mmlongbench_manifest.py`): same pattern,
  messier sourcing — it's vision-language (rows can carry images) and has no single community mirror
  covering all 16 tasks, so each task maps individually to whatever public source has it (original
  upstream datasets, or two re-uploads with images embedded as HF `Image` features). Read the module
  docstring before touching `CATEGORIES` — it documents per-task caveats (merged task variants,
  unavailable tasks, text-only sources). Embedded images use HF's `{"src","height","width"}` shape;
  `shared.js`'s `looksLikeImage()`/`renderImage()` render them as thumbnails, and `truncate()` leaves
  image cells untouched.

- **Dolci** (`fetch_dolci_source_samples_duckdb.py` → `build_dolci_source_manifest.py`): Ai2's
  post-training data suite for Olmo 3. Currently covers `allenai/Dolci-Instruct-SFT` (the SFT-stage
  mixture, 2.15M rows), broken into one category per upstream sub-source rather than one category for
  the whole mixture — every row carries a `source_dataset` column identifying which of ~22 blended
  sources (FLAN v2, Tulu 3 personas, Aya, newly-authored Dolci sources, etc.) it came from. Sampling
  is done by downloading Dolci-Instruct-SFT's auto-converted parquet shards once (~3GB, cached
  outside the repo — see `DOLCI_PARQUET_CACHE` in the script) and querying them with local
  **DuckDB** (`pip install duckdb`, not stdlib), not the `datasets-server` `/filter` endpoint, which
  proved unreliable (its query-time index warms up flakily across backend replicas, causing frequent
  transient 500s). The older
  `fetch_dolci_samples.py` → `build_dolci_manifest.py` pair (whole-repo-as-one-category, via
  `/filter`) is kept as a fallback/reference. All fetch scripts read an optional `HF_TOKEN` env var
  and send it as a Bearer token if set, but don't require it for public datasets.

To add a Dolci sibling stage (DPO, RL, Think/RL-Zero variants), add its `repo_id`/source breakdown
and rerun the fetch + build scripts.

## Training study architecture (`training/`)

`training/index.html` is a hand-curated `<table>`, not scraped — training-time figures (throughput,
wall-clock, token counts) are reported inconsistently enough across model releases that a scraper
can't reliably fill every cell.

**Columns:** Model, URL, Size, Training stage, GPU setup, Training throughput, Est. training
GPU-hours, # of tokens.

**Methodology:** quote figures directly from the source when stated. When GPU-hours aren't given
directly, compute `GPU count × wall-clock training time`, mark the cell "computed" via a
`<span class="cell-note">`, and cross-check against any partial figures the source does give.

To add a row: edit the `<tbody>` directly, keep column order, use the `mono` cell class for
numeric/technical values, and add a `cell-note` under any figure needing a caveat (units,
reported-vs-computed, assumptions).
