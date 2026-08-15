# Development

The site is a static, dependency-free hub (`index.html` + `style.css` at the
repo root) linking out to one page per survey. Each survey lives in its own
top-level directory and owns its data/pipeline.

## Dataset Survey (`datasets/`)

`datasets/index.html` is a small chooser page (same `.survey-grid` card
pattern as the site root) linking to one full page per source:
`datasets/nemotron/`, `datasets/helmet/`, and `datasets/mmlongbench/`. Each
source has its own URL, so
browsing one never means scrolling through the other first — this used to be
a single page with both sources stacked vertically, which was bad UX once
Nemotron alone had 100+ dataset cards.

Rendering is shared via `datasets/shared.js` (`DatasetSurvey.init(config)`):
category nav, dataset cards, the sample-row modal, everything except the
per-source manifest URL and intro copy, which each subpage's own inline
`<script>` passes in. `config.manifestUrl` points at that source's manifest
JSON; `sample_file` in each dataset entry is a path *relative to the subpage*
(e.g. `../data/samples/x.json`, since `nemotron/` and `helmet/` sit one level
below `datasets/data/`) that resolves via `fetch()` directly.

To add a new source: write a script that produces a manifest in the same
shape (`{categories: [{title, description, url, datasets: [{repo_id, url,
description, sample_status, sample_file, ...}]}]}`), add a subpage
(`datasets/<name>/index.html`, copy an existing one and change the `init()`
call), and add a card for it to `datasets/index.html`.

### Nemotron (NVIDIA dataset collections)

1. **`datasets/scripts/fetch_collections.py`** — queries the Hugging Face Hub
   [collections API](https://huggingface.co/docs/hub/api#collections) for
   every `nvidia`-owned collection whose title mentions "nemotron", keeps the
   ones containing dataset repos, and writes `datasets/data/categories.json`.
2. **`datasets/scripts/fetch_dataset_metadata.py`** — pulls license,
   downloads, likes, and a short description for each unique dataset repo,
   into `datasets/data/dataset_metadata.json`.
3. **`datasets/scripts/sample_datasets.py`** — for each unique dataset, asks
   the [datasets-server rows API](https://huggingface.co/docs/dataset-viewer)
   for the first 16 rows of its default/train split. This reads from a
   pre-built Parquet export server-side, so it works even for multi-terabyte
   datasets without downloading them. Results are cached per-dataset under
   `datasets/data/samples/`; gated or viewer-less datasets are recorded as
   such rather than silently skipped.
4. **`datasets/scripts/build_manifest.py`** — combines the three into
   `datasets/data/manifest.json`, the single file the frontend fetches on
   load. Sample rows stay in their own per-dataset files and are fetched
   lazily when a user clicks "View 16 examples", so the initial page load
   stays light.

Regenerate with:

```bash
cd datasets
python3 scripts/fetch_collections.py
python3 scripts/fetch_dataset_metadata.py
python3 scripts/sample_datasets.py   # re-run to retry only failed datasets
python3 scripts/build_manifest.py
```

No API key needed for public datasets. Gated datasets need `huggingface-cli
login` / an `HF_TOKEN` env var to sample — this project currently skips those
and marks them as gated in the UI rather than authenticating.

A dataset can legitimately appear in more than one category — NVIDIA's own
collections overlap (e.g. a math dataset shows up in both "Math & Reasoning"
and the "Post-Training-v3" blend collection). This mirrors how the source
collections are organized, not a bug.

### HELMET (long-context eval benchmark, arXiv:2410.02694)

The category → dataset → HF-repo mapping is hand-curated in
`datasets/scripts/fetch_helmet_samples.py` (`CATEGORIES`), built from Table 3
of the paper. The official `princeton-nlp/HELMET` dataset repo ships its
preprocessed eval files as one 34GB tarball with no per-task dataset-server
viewer, so each task's file is instead read from a community re-upload by
[xiaoyuanliu](https://huggingface.co/xiaoyuanliu) — verified, not assumed:
each repo's fields were inspected (e.g. RULER MK Needle vs. MK UUID were
told apart via the `type_needle_v` field) before mapping it to a task. If
Princeton NLP ever ships per-task viewer-enabled repos, swap the `repo_id`
values in `CATEGORIES` to point at those instead.

Because HELMET rows contain full long-context prompts (up to ~128K tokens),
`fetch_helmet_samples.py` truncates any string field over 4,000 characters
when saving — otherwise a 16-row sample file would be tens of MB. Samples are
also a true random draw (fixed seed 42, via the `datasets-server` `/size`
endpoint plus per-index row fetches), not just the first 16 rows.

Regenerate with:

```bash
cd datasets
python3 scripts/fetch_helmet_samples.py     # re-run to retry only failed tasks
python3 scripts/build_helmet_manifest.py
```

### MMLongBench (long-context vision-language eval benchmark, arXiv:2505.10610)

Same idea as HELMET — a hand-curated `CATEGORIES` table in
`datasets/scripts/fetch_mmlongbench_samples.py`, built from Table 2 of the
paper — but the sourcing story is messier, because MMLongBench is a
*vision-language* benchmark (rows can carry images) and its official repo
(`ZhaoweiWang/MMLongBench`) has the same "34GB tarball, no per-task viewer"
problem as HELMET's official repo, except worse: images are referenced by
relative file path into separate per-category tarballs rather than embedded,
so even a working viewer wouldn't show them inline.

There's no single community mirror covering all 16 tasks the way
xiaoyuanliu's did for HELMET, so each task is mapped individually to the
best public source found — usually the original upstream benchmark
MMLongBench itself draws on (Stanford Cars, Food101, SUN397, GovReport,
Multi-LexSum — the last one reuses the exact same HELMET mirror file, since
it's the same underlying dataset), or, for two tasks, a community re-upload
with images properly embedded as an HF `Image` feature
([shrekwang](https://huggingface.co/shrekwang)'s `bam_mmlongbench-doc` and
`bam_longdocurl`). Full detail and caveats are documented in the module
docstring at the top of `fetch_mmlongbench_samples.py` — read that before
changing the mapping. Notably:

- **VH-Single/VH-Multi** and **MM-NIAH's Ret/Count/Reason** are each shown
  as one merged card, not split like the paper's table — every public
  mirror checked only exposed one blended task variant via the viewer
  (confirmed by sampling many offsets across each dataset and finding no
  variation in the field that should distinguish them).
- **InfoSeek** and **SlideVQA** have no working public source (schema
  errors / gated / persistent server errors across every candidate tried)
  and are marked unavailable, same as any gated/broken dataset elsewhere on
  the site.
- Several sources (ViQuAE, Visual Haystack, MM-NIAH) reference images by
  filename into external corpora (Wikipedia Commons, COCO, OBELICS) rather
  than embedding them, so those samples are text-only.

Embedded images arrive as HF's standard `{"src", "height", "width"}` cell
shape; `shared.js`'s `renderValue()` detects that shape (and lists of it)
and renders actual `<img>` thumbnails instead of falling through to a JSON
dump — see `looksLikeImage()`/`renderImage()` in `datasets/shared.js`.
`fetch_mmlongbench_samples.py`'s `truncate()` leaves image cells untouched
(only strings and long lists are truncated) so those thumbnails still work
after sampling.

Regenerate with:

```bash
cd datasets
python3 scripts/fetch_mmlongbench_samples.py   # re-run to retry only failed tasks
python3 scripts/build_mmlongbench_manifest.py
```

### Dolci (Ai2's Olmo 3 post-training data)

Same idea as HELMET/MMLongBench — a hand-curated `CATEGORIES` table, this time
in `datasets/scripts/fetch_dolci_samples.py` — but simpler sourcing: every
Dolci dataset is a first-party, viewer-enabled `allenai/` repo, so no mirror
hunting is needed, just the `datasets-server` rows API directly.

Dolci is Ai2's post-training data suite for Olmo 3: each model variant
(Instruct, Think, RL-Zero) gets its own SFT / DPO / RL dataset (e.g.
`Dolci-Instruct-SFT`, `Dolci-Instruct-DPO`, `Dolci-Instruct-RL`,
`Dolci-Think-SFT-{7B,32B}`, `Dolci-RL-Zero-{Math,Code,IF,General}-7B`, ...).
This currently covers only `allenai/Dolci-Instruct-SFT` — the SFT-stage data
(2,152,112 examples) for the Olmo 3 Instruct models, confirmed against the
[Olmo 3.1 32B Instruct model card](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct)
(SFT → DPO → RLVR, on Dolci-Instruct-SFT / -DPO / -RL respectively). To add a
sibling stage, add its `repo_id` to `CATEGORIES` (a new category, or a new
entry in the existing one) and rerun both scripts below.

Regenerate with:

```bash
cd datasets
python3 scripts/fetch_dolci_samples.py   # re-run to retry only failed datasets
python3 scripts/build_dolci_manifest.py
```

## Training Config & Time Survey (`training/`)

`training/index.html` is currently a hand-curated table (no scraper) — each
row is added by reading the model's own repo/paper and filling in the columns
directly in the HTML. This is intentional: training-time figures are reported
inconsistently across projects (some give throughput, some give wall-clock
time, some give neither), so an automated scraper can't reliably fill every
cell the way the Hugging Face APIs can for dataset metadata.

**Columns:** Model, URL, Size, Training stage, GPU setup, Training
throughput, Est. training GPU-hours, # of tokens.

**Methodology:** quote figures directly from the source when available. When
GPU-hours aren't stated directly, compute `GPU count × wall-clock training
time` and mark the cell "computed" (see the `.cell-note` under that cell) so
readers can tell reported numbers from derived ones. Cross-check computed
numbers against any partial figures the source does give (e.g. TinyLlama's
paper states 3,456 GPU-hours for a 300B-token subset, which scales linearly
to the ~34,560 GPU-hours computed for the full 3T-token run).

### Adding a row

Edit the `<tbody>` in `training/index.html` directly. Keep the same column
order, use the `mono` cell class for numeric/technical values, and add a
`<span class="cell-note">` under any figure that needs a caveat (units,
whether it's reported vs. computed, source assumptions).
