# Project Guidelines

## Code Style
- Follow the existing Python style in [src/StanceDetector.py](src/StanceDetector.py): explicit method names, docstring-first methods, and minimal side effects outside the topic record.
- Keep comments concise and only where logic is non-obvious.
- Preserve public method names/signatures in `StanceDetector` unless explicitly requested.

## Architecture
- Main pipeline lives in [src/StanceDetector.py](src/StanceDetector.py):
  1. `filter_speeches`
  2. `classify_filtered_sentences`
  3. `summarize_all_sentences`
  4. `generate_anchors`
  5. `compute_embeddings` / `compute_umap_embeddings`
  6. plotting/evaluation methods
- Topic metadata (`keywords`, `policyarea`) is defined in [src/json_generation.py](src/json_generation.py).
- Processed outputs are stored in [data/processed](data/processed).
- Interactive experimentation is notebook-driven in [notebooks/stance_detection.ipynb](notebooks/stance_detection.ipynb).

## Build and Test
- No formal build/test automation is currently defined (no `requirements.txt`, `pyproject.toml`, or CI config).
- Typical local workflow:
  - Activate environment: `source .venv/bin/activate`
  - (If needed) regenerate records: `python src/json_generation.py`
  - Run notebook workflow: `jupyter notebook notebooks/stance_detection.ipynb`
- If adding tests, use `pytest` with small deterministic fixtures in `data/` and document the command here.

## Conventions
- `self.__record` is the source of truth for per-topic intermediate artifacts; store new per-topic outputs there.
- Keep alignment between embedding arrays and speaker metadata (see `df_embeddings_speaker` usage in [src/StanceDetector.py](src/StanceDetector.py)).
- Prefer deterministic runs where possible by using `random_seed` when calling models.
- When adding a topic, define both `keywords` and `policyarea` and keep names consistent across scripts/notebooks.

## Environment Requirements and Pitfalls
- Ollama must be running locally for summarization/anchor generation (`http://localhost:11434`).
- Required local models may include `gemma3`, `qwen3`, `deepseek-r1:8b`, `ministral-3:8b` depending on experiment scripts.
- Hugging Face model access is required for `andreacristiano/stancedetection` in `SetFitModel.from_pretrained`.
- Embedding computation with `Qwen/Qwen3-Embedding-0.6B` can be slow on CPU.

## Reference Docs
- Data/source notes: [sources.md](sources.md)
- Topic configuration example: [src/json_generation.py](src/json_generation.py)
- Utility helpers and ROUGE evaluation: [src/utils.py](src/utils.py)
