# LLM Dataset Explorer

A browsable catalog of the datasets behind open LLM model families — grouped the
way the model creator groups them, linked back to the original Hugging Face
repos, with real sampled rows so you can see what's actually inside without
downloading anything.

**Currently featuring: [NVIDIA Nemotron](https://huggingface.co/nvidia/collections?search=nemotron)**

Live site: `https://<your-username>.github.io/llm-dataset-explorer/` (enable
GitHub Pages on this repo, serving from the root of `main`).

## How it works

1. **`scripts/fetch_collections.py`** — queries the Hugging Face Hub
   [collections API](https://huggingface.co/docs/hub/api#collections) for
   every `nvidia`-owned collection whose title mentions "nemotron", keeps the
   ones containing dataset repos, and writes `data/categories.json`.
2. **`scripts/fetch_dataset_metadata.py`** — pulls license, downloads, likes,
   and a short description for each unique dataset repo, into
   `data/dataset_metadata.json`.
3. **`scripts/sample_datasets.py`** — for each unique dataset, asks the
   [datasets-server rows API](https://huggingface.co/docs/dataset-viewer) for
   the first 16 rows of its default/train split. This reads from a pre-built
   Parquet export server-side, so it works even for multi-terabyte datasets
   without downloading them. Results are cached per-dataset under
   `data/samples/`; gated or viewer-less datasets are recorded as such rather
   than silently skipped.
4. **`scripts/build_manifest.py`** — combines the three into
   `data/manifest.json`, the single file the frontend fetches on load. Sample
   rows stay in their own per-dataset files and are fetched lazily when a
   user clicks "View 16 examples", so the initial page load stays light.
5. **`index.html` / `style.css` / `app.js`** — a static, dependency-free
   frontend: category sections, dataset cards with license/download badges, a
   name filter, and a modal that renders sampled rows generically (handles
   long text, nested chat/JSON structures, and embedding vectors).

## Regenerating the data

```bash
python3 scripts/fetch_collections.py
python3 scripts/fetch_dataset_metadata.py
python3 scripts/sample_datasets.py   # re-run to retry only failed datasets
python3 scripts/build_manifest.py
```

No API key needed for public datasets. Gated datasets need `huggingface-cli
login` / an `HF_TOKEN` env var to sample — this project currently skips those
and marks them as gated in the UI rather than authenticating.

## Notes on the data

- A dataset can legitimately appear in more than one category — NVIDIA's own
  collections overlap (e.g. a math dataset shows up in both "Math &
  Reasoning" and the "Post-Training-v3" blend collection). This mirrors how
  the source collections are organized, not a bug.
- Sample rows belong to their original authors and dataset licenses. Each
  dataset card links back to its Hugging Face page — check there before
  reusing anything beyond the 16-row preview shown here.

## Adding another model family

The pipeline isn't Nemotron-specific beyond the `owner=nvidia` /
`"nemotron"` filter in `fetch_collections.py`. To add another family, adapt
that filter (or point it at a different org/search term), regenerate the
data files into a separate `data/<family>/` directory, and extend the
frontend to switch between families.
