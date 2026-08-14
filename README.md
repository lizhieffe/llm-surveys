# LLM Surveys

[![View Live Site](https://img.shields.io/badge/%F0%9F%94%8D_View_Live_Site-lizhieffe.github.io%2Fllm--surveys-4c8f2f?style=for-the-badge)](https://lizhieffe.github.io/llm-surveys/)

Grounded surveys of how open LLMs are actually built, each figure linked back to
its primary source.

- **[Nemotron Dataset Survey](datasets/)** — every dataset collection NVIDIA has
  published for the Nemotron model family, with 16 sampled rows per dataset
  pulled live from the Hugging Face dataset viewer.
- **[Training Config & Time Survey](training/)** — a model-by-model table of
  training hardware, throughput, token counts, and estimated GPU-hours.
  Starts with TinyLlama.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for how each survey's pipeline
works and how to extend it.
