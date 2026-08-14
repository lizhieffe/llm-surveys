# Development

The site is a static, dependency-free hub (`index.html` + `style.css` at the
repo root) linking out to one page per survey. Each survey lives in its own
top-level directory and owns its data/pipeline.

## Dataset Survey (`datasets/`)

One page, multiple sources: `datasets/app.js` has a `SOURCES` array, each
entry pointing at its own manifest file (`data/manifest.json` for Nemotron,
`data/helmet-manifest.json` for HELMET). Each source renders as its own
self-contained block (intro card, search, category nav, category sections) on
the same page — the category/dataset card/modal rendering code is shared. To
add a new source, add an entry to `SOURCES` and build a matching manifest
(same shape: `{categories: [{title, description, url, datasets: [{repo_id,
url, description, sample_status, sample_file, ...}]}]}`, with `sample_file`
as a path relative to `datasets/` that resolves via `fetch()`).

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
4. **`datasets/scripts/build_manifest.py`** — combines the three into
   `datasets/data/manifest.json`.

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

### Adding another source

Nothing about the pipeline is Nemotron- or HELMET-specific beyond the two
scripts above. To add a new source: write a script that produces a manifest
in the same shape (see the `SOURCES` note at the top of this section), point
its dataset cards' `sample_file` at wherever you save the per-dataset sample
JSON, and add an entry to `SOURCES` in `datasets/app.js`.

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
