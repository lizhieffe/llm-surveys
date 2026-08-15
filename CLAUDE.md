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

Rendering is shared via `datasets/shared.js` (`DatasetSurvey.init(config)`): nav, headers, dataset
cards, and the sample-row modal are all common code. Each subpage's inline `<script>` only supplies
`config.manifestUrl` (that source's manifest JSON) and intro copy. A manifest is either **2 layers**
(flat: category → dataset cards — Nemotron, HELMET, MMLongBench) or **3 layers** (grouped: group →
category → dataset card — Dolci, where "group" = one Dolci dataset repo and "category" = one native
sub-source within it). `shared.js` picks the shape from whether the manifest has a top-level
`groups` key; the layer below category (the dataset card, and the sample modal it opens to show 16
rows) is identical code either way, so a dataset needs 3 layers only when its categories themselves
need a further umbrella grouping — most sources stay 2-layer.

```
// flat (2 layers)
{categories: [{title, description, url, stage?, datasets: [{repo_id, url, description, license,
  downloads, likes, sample_status, sample_file, ...}]}]}

// grouped (3 layers)
{groups: [{title, description, url, metric, categories: [{title, description, url,
  datasets: [{repo_id, url, description, license, downloads, likes, sample_status,
  sample_file, ...}]}]}]}
```

`stage` (flat manifests only, currently just Nemotron) is an optional per-category badge string
(`"Pre-training"` / `"Post-training"` / `"Mixed"`) rendered next to the category title when present;
omit it rather than guessing when a source gives no clear signal either way.

`sample_file` paths are relative to the subpage (e.g. `../data/samples/x.json`), fetched lazily on
"View N examples" click so initial page load stays light. Long text fields and long lists are
truncated when samples are written (not at render time), so sample files stay small.

To add a new source: write a script producing a manifest in the shape above, add
`datasets/<name>/index.html` (copy an existing subpage, change the `init()` config), and add a card
to `datasets/index.html`.

**Categorization rule:** if a dataset has documented native sub-sets (a blend of named upstream
sources, as most Dolci/post-training datasets are), each sub-set becomes its own category — never one
category for the whole dataset — and each gets its own 16 sampled rows. Find the breakdown from the
HF dataset card, its linked paper/blog post, or by searching the web; don't guess. Then verify it
against the actual data before building anything: check the row schema (`datasets-server` `/rows`)
for a source-identifying column (name and granularity vary — e.g. Dolci-Instruct-SFT's
`source_dataset` uses clean human labels that match the card's breakdown directly, while
Dolci-Think-SFT-7B's `dataset_source` uses raw upstream repo strings at finer granularity, needing a
many-to-one mapping to the card's named categories), and confirm counts sum to the documented total.
Don't trust `datasets-server`'s `/statistics` endpoint for this on large datasets — it can silently
return `partial: true` (stats from only the first chunk of rows), which looks complete but isn't; get
true full-dataset counts via a local `GROUP BY` (see the Dolci entry below).

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
  overlapping NVIDIA collections). Nemotron's ecosystem mixes pre-training and post-training data
  under one brand, so `build_manifest.py`'s `classify_stage()` tags each category with an optional
  `"Pre-training"` / `"Post-training"` / `"Mixed"` `stage` field (`shared.js` renders it as a badge
  next to the category title when present) — but only when NVIDIA's *own* collection description
  contains explicit textual evidence (e.g. "pre-training datasets", "SFT", "reward model", "RLHF");
  categories with no such wording are left unbadged rather than guessed. Two manual overrides exist
  where the evidence lives one level up from the category's own text (see `_STAGE_OVERRIDES` in the
  script) — everything else must earn its badge from its own description.

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

- **Dolci** (`build_dolci_source_manifest.py`, combining per-stage fetch scripts via a `STAGES`
  list): Ai2's post-training data suite for Olmo 3 — the site's one **3-layer (grouped)** manifest.
  Covers all six SFT/DPO/RL stages for both the Instruct and Think 7B models (everything except the
  RL-Zero family); each stage is its own dataset repo and becomes one layer-1 **group** (title,
  description, `url` pointing at that stage's own HF page, a `metric` badge summing its categories'
  prompt counts), with that stage's native sub-sources as plain, unprefixed layer-2 **categories**
  inside it (no more cross-stage title prefixing or slug-suffixing needed now that stages don't
  share a flat list):
  - `fetch_dolci_source_samples_duckdb.py` → `allenai/Dolci-Instruct-SFT` (2.15M rows). Its
    `source_dataset` column already uses clean human labels matching the card's ~22 categories
    1:1.
  - `fetch_dolci_think_source_samples_duckdb.py` → `allenai/Dolci-Think-SFT-7B` (2.27M rows, 36GB
    across 156 shards — much bigger, since it carries full reasoning traces). Its `dataset_source`
    column uses raw upstream-repo-style strings at *finer* granularity than the card's 13
    categories (e.g. the card's "OpenThoughts 3" is 3 separate `dataset_source` values that sum to
    its exact stated count) — see `DOLCI_THINK_SOURCES` for the verified many-to-one mapping.
  - `fetch_dolci_think_dpo_source_samples_duckdb.py` → `allenai/Dolci-Think-DPO-7B` (150K
    preference pairs, 7 shards, ~1.4GB — much smaller than either SFT mixture). Row shape is
    `prompt`/`chosen`/`rejected` (a preference pair), not a single `messages` list. Its card gives
    **no source breakdown at all** — the 24 `DOLCI_THINK_DPO_SOURCES` categories come entirely
    from a local `GROUP BY dataset_source` (which happened to sum exactly to 150,000 with zero
    unmatched rows, unlike Think-SFT-7B's identity-prompts gap), with each raw source string's
    upstream repo found via Hugging Face search rather than assumed.
  - `fetch_dolci_instruct_dpo_source_samples_duckdb.py` → `allenai/Dolci-Instruct-DPO` (260K
    preference pairs, 4 shards). This one has **no source-identifying column of any kind** — rows
    aren't tagged with an upstream prompt collection at all. Its card instead describes a blend of
    *preference-construction methods*, and the row schema carries that split directly via
    `preference_type` (4 values: an LLM-judged UltraFeedback-style pipeline, Delta Learning, and
    two multiturn variants) — confirmed by `GROUP BY preference_type` matching the card's
    (rounded) per-method counts almost exactly. When a dataset has no source column, its native
    categorization axis is whatever column *does* vary meaningfully — check the schema for
    candidates before concluding there's no way to break it down.
  - `fetch_dolci_think_rl_source_samples_duckdb.py` → `allenai/Dolci-Think-RL-7B` (102K RLVR
    prompts, 9 shards). Its card gives **two breakdowns at once** — a coarse 4-way "Grouped Mixes"
    split and a fine 13-way "Original Dataset Contribution" split, both summing exactly to the
    total. A `GROUP BY` on each candidate schema column (`dataset_source`, `original_dataset`)
    showed which maps to which: `dataset_source` == the coarse split, `original_dataset` == the
    fine one (again at slightly finer granularity than the card — one category is 3 raw values).
    When a card offers multiple breakdowns, use the schema to find which column backs each one,
    then take the finer one per the categorization rule. This dataset's rows also carry large
    tokenized fields (`input_ids`, `attention_mask`, `labels`) meaningless without the training
    tokenizer — the fetch script's `SELECT` clause explicitly excludes them rather than truncating
    them, since even truncated they'd add noise with zero readability benefit.
  - `fetch_dolci_instruct_rl_source_samples_duckdb.py` → `allenai/Dolci-Instruct-RL` (170K RLVR
    prompts, 3 shards). Also gives two breakdowns, but unlike Think-RL-7B **they don't agree in
    coverage**: the "Grouped Mixes" table (8 mixes) sums exactly to the total and its column
    (`dataset_source`) has no nulls, while the "Original Dataset Contribution" table (4 rows) sums
    to only half the total — its column (`original_dataset`) is `NULL` for the 5 Math/Code mixes
    entirely, and only breaks down one mix ("General RLVR Mix") into its real 3 components. The fix
    was per-category, not per-dataset: 7 of the 8 `DOLCI_INSTRUCT_RL_SOURCES` categories filter on
    `dataset_source` directly, but the 3 "General RLVR Mix" sub-parts filter on `original_dataset`
    instead, since that's the only column with real data for them. Lesson: a dataset's finest usable
    column can vary *per row*, not just per dataset — check `NULL` coverage before trusting a more
    granular column everywhere.

  All six fetch scripts work the same way: download the dataset's auto-converted parquet shards
  once (cached outside the repo — see `DOLCI_PARQUET_CACHE`/`DOLCI_THINK_PARQUET_CACHE`/
  `DOLCI_THINK_DPO_PARQUET_CACHE`/`DOLCI_INSTRUCT_DPO_PARQUET_CACHE`/`DOLCI_THINK_RL_PARQUET_CACHE`/
  `DOLCI_INSTRUCT_RL_PARQUET_CACHE`) and query them with local **DuckDB** (`pip install duckdb`, not
  stdlib — if `import duckdb` fails, check you're not inside another project's venv; it may only be
  installed against a specific `python3`, e.g. a miniconda base install, not whichever `python3` is
  first on `PATH`), rather than the `datasets-server` `/filter` endpoint, which proved unreliable
  (its query-time index warms up flakily across backend replicas, causing frequent transient 500s).
  The older `fetch_dolci_samples.py` → `build_dolci_manifest.py` pair (whole-repo-as-one-category,
  via `/filter`) is kept as a fallback/reference. All fetch scripts read an optional `HF_TOKEN` env
  var and send it as a Bearer token if set, but don't require it for public datasets.

  **Before trusting *any* dataset card's source breakdown, verify it against the real data**: run a
  full `GROUP BY <source-column>` over the downloaded parquet (not `datasets-server`'s
  `/statistics` endpoint — see the categorization rule above) and check counts sum to what the card
  claims — or, if the card gives no breakdown at all (Think-DPO-7B, Instruct-DPO), let the
  `GROUP BY` itself *be* the category list. This caught a real discrepancy in Dolci-Think-SFT-7B:
  the card's "Olmo Identity Prompts" (58 rows) has no distinguishable `dataset_source` tag in the
  actual release, so that category is marked `no_rows` rather than guessed at.

To add a Dolci sibling stage (the RL-Zero family): write a
`fetch_dolci_<stage>_source_samples_duckdb.py` (must expose `DATASET`, a `*_SOURCES` list, and
`safe_filename()`) following an existing one as a template, then add one row to the `STAGES` list
at the top of `build_dolci_source_manifest.py`.

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
