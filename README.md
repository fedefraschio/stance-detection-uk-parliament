# Controversy Axes in UK Parliamentary Debates: An LLM-driven Approach

An NLP pipeline for automatic stance detection and controversy analysis applied to UK parliamentary debates. This project uses large language models to identify political positions along contested issues and validate them through embedding-based projections.

## Overview

This work presents a complete methodology for:
1. **Topic filtering** — Selecting relevant speeches by keyword and policy area
2. **Opinion detection** — Identifying opinion-bearing sentences via fine-tuned classification
3. **Stance summarization** — Generating per-politician stance summaries using local LLMs
4. **Axis generation** — Automatically identifying controversy axes and stance anchors
5. **Embedding projection** — Projecting party positions onto controversy axes
6. **Validation** — Comparing predicted orderings against LLM-generated gold standards

The approach is inspired by the [KOKKAI DOC framework](https://arxiv.org/abs/2505.07118) and evaluated on multiple political topics from the [ParlEE dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VOPK0E).

## Dataset

- **Source**: ParlEE Plenary Speeches V2 (2009–2019)
- **Size**: 15.1 million sentence-level speeches from six EU parliamentary chambers
- **Used for**: UK parliamentary debates analysis
- **Time periods analyzed**: 
  - Gaza/Israeli-Palestinian conflict: 2011–2014
  - Climate change: 2014

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended for embedding computation)
- Ollama with local LLM support (for stance summarization)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd stance-detection-eu-parliaments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama service (if using local LLM)
ollama serve
```

## Usage

The main analysis is conducted in the Jupyter notebook:

```bash
jupyter notebook notebooks/stance_detection.ipynb
```

### Key Steps

1. **Data Loading**: Loads UK parliamentary debates from HuggingFace
2. **Speech Filtering**: Filters by topic keywords and policy area codes
3. **Pipeline Execution**: Runs classification, summarization, and anchor generation
4. **Evaluation**: Projects parties onto controversy axes and validates against gold standards

### Example: Analyzing a New Topic

To analyze a new topic:

1. Add keywords and policy area codes to `data/external/records.json`
2. Define `topic` and `years` variables in the notebook
3. Run the filtering → classification → summarization pipeline
4. Generate anchors or provide manual anchor definitions
5. Compute embeddings and project onto controversy axes
6. Validate using LLM-as-a-judge approach

## Data Structure

```
data/
├── external/
│   └── records.json          # Topic definitions (keywords, policy areas)
├── raw/
│   └── ParlEE_UK_plenary_speeches.csv  # Raw dataset
└── processed/
    ├── summarizations_Gaza.csv              # Politician stance summaries
    ├── anchors_Gaza.json                    # Controversy axes definitions
    ├── summarizations_climate change.csv
    └── anchors_climate change.json
```

## Methodology

### Opinion Detection
Fine-tuned SetFit classification model identifies opinion-bearing sentences from objective reporting.

### Stance Summarization
Local LLM (Qwen 3:8b) generates concise stance summaries from opinion sentences, preserving speaker positions.

### Anchor Generation
LLM-generated PRO/CON anchors define the poles of each controversy axis, automatically identified from aggregated summaries.

### Projection & Scoring
Party stances are embedded and projected onto controversy axes via scalar projection, producing controversy scores.

### Validation
LLM-as-a-judge generates gold standard party orderings based on general political knowledge. Predicted orderings are compared using:
- **Spearman's ρ** — Rank correlation coefficient
- **Kendall's τ** — Pair concordance metric
- **LCS ratio** — Longest common subsequence

## Results Summary

### Gaza (2011–2014)

**General Issue**: Primary obstacle to peace in Israeli-Palestinian conflict
- 7 sub-topics analyzed
- Spearman's ρ range: 0.5–1.0 (most strong correlations)
- Average agreement: 85%+ with gold standard on general issue

### Climate Change (2014)

**General Issue**: Role of government in UK's fossil fuel transition
- 5 sub-topics analyzed
- Spearman's ρ range: 0.4–0.9
- Strong performance on broad framing, degraded on narrow sub-issues

## Limitations & Future Work

1. **Summarization Granularity** — Topic-level summaries limit sub-topic accuracy
2. **Sparse Signal** — Parties with few speakers produce unreliable estimates
3. **Statistical Power** — Limited number of parties reduces correlation metric reliability
4. **Gold Standard Bias** — LLM rankings reflect general knowledge, not period-specific data
5. **Anchor Quality** — No automatic filtering of poorly-separated anchor pairs

**Recommendations**:
- Implement minimum speaker threshold per party
- Generate sub-topic-level summaries
- Evaluate at politician level for increased power
- Multi-run gold standard generation with median aggregation
- Automatic anchor quality gates (cosine similarity threshold)

## Project Structure

```
src/
├── StanceDetector.py    # Main pipeline class
├── utils.py             # Helper functions
└── json_generation.py   # Data processing utilities

notebooks/
├── stance_detection.ipynb    # Main analysis notebook
├── 01_summarization.ipynb    # Stance summarization exploration
├── 02_preprocessing.ipynb    # Data preprocessing
└── 03_classification.ipynb   # Opinion detection model training
```

## Key Dependencies

- `pandas`, `numpy` — Data manipulation
- `sentence-transformers` — Embedding computation
- `setfit` — Opinion classification
- `umap-learn` — 2D visualization
- `scipy` — Rank correlation metrics
- `matplotlib` — Plotting
- `ollama` — Local LLM inference

## References

[1] Sylvester, C., Greene, Z., Ershova, A., Khokhlova, A., & Yordanova, N. (2023). ParlEE plenary speeches V2 dataset: Annotated full-text of 15.1 million sentence-level plenary speeches of six EU legislative chambers. Harvard Dataverse. https://doi.org/10.7910/DVN/VOPK0E

[2] Kato, K., & Cochrane, C. (2025). KOKKAI DOC: An LLM-driven framework for scaling parliamentary representatives. arXiv:2505.07118

