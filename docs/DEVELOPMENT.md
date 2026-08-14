# Development

The site is a static, dependency-free hub (`index.html` + `style.css` at the
repo root) linking out to one page per survey. Each survey lives in its own
top-level directory and owns its data/pipeline.

## Nemotron Dataset Survey (`datasets/`)

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
5. **`datasets/index.html` / `datasets/style.css` / `datasets/app.js`** — a
   static frontend: category sections, dataset cards with license/download
   badges, a name filter, and a modal that renders sampled rows generically
   (handles long text, nested chat/JSON structures, and embedding vectors).

### Regenerating the dataset survey data

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

### Notes on the data

- A dataset can legitimately appear in more than one category — NVIDIA's own
  collections overlap (e.g. a math dataset shows up in both "Math &
  Reasoning" and the "Post-Training-v3" blend collection). This mirrors how
  the source collections are organized, not a bug.
- Sample rows belong to their original authors and dataset licenses. Each
  dataset card links back to its Hugging Face page — check there before
  reusing anything beyond the 16-row preview shown here.

### Adding another model family

The pipeline isn't Nemotron-specific beyond the `owner=nvidia` /
`"nemotron"` filter in `fetch_collections.py`. To add another family, adapt
that filter (or point it at a different org/search term) and regenerate the
data files — either into `datasets/data/` if replacing Nemotron, or into a
new sibling survey directory if adding it alongside.

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
