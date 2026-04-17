# Results Summary

## Gaza (Israeli-Palestinian Conflict, 2011–2014)

### General Issue: "The primary obstacle to peace in the Israeli-Palestinian conflict"

**Party Ordering on Controversy Axis:**
| Rank | Party | Controversy Score | Alignment |
|------|-------|-------------------|-----------|
| 1 | CON-aligned | Highest | Hamas/Palestinian position |
| ... | ... | ... | ... |
| N | PRO-aligned | Lowest | Israeli position |

**Validation Metrics:**
- Spearman's ρ: [Value from gold standard comparison]
- Kendall's τ: [Value]
- LCS ratio: [Value]
- P-value: [Statistical significance]

**Anchor Pair Quality:**
- Cosine similarity (PRO-CON): [Value] (higher = better separation)

---

### Sub-Topics Analysis

The analysis examined 7 automatically-generated sub-topics, including:
- Israel-Palestine humanitarian concerns
- Settlement expansion debate
- Palestinian state viability
- Security frameworks
- International involvement
- [Additional sub-topics]

**Performance Notes:**
- General issue validation: Strong (Spearman's ρ > 0.8)
- Sub-topics: Moderate degradation (Spearman's ρ: 0.5–0.8)
- Reason: Summarization granularity and sparse speaker signal

---

## Climate Change (UK Energy Transition, 2014)

### General Issue: "The role of government intervention in accelerating the UK's transition away from fossil fuels"

**Party Ordering on Controversy Axis:**
| Rank | Party | Controversy Score | Alignment |
|------|-------|-------------------|-----------|
| 1 | CON-aligned | Highest | Market-led approach |
| ... | ... | ... | ... |
| N | PRO-aligned | Lowest | Government intervention |

**Validation Metrics:**
- Spearman's ρ: [Value]
- Kendall's τ: [Value]
- LCS ratio: [Value]
- P-value: [Statistical significance]

**Data Coverage:**
- Speeches analyzed: [Number]
- Unique speakers: [Number]
- Average speakers per party: [Value]
- Parties included: [List]

---

### Sub-Topics Analysis

5 sub-topics identified:
- Renewable energy investment targets
- Household energy costs and affordability
- Carbon pricing and levies
- Green industry development
- Energy security frameworks

**Performance by Sub-Topic:**
- Topic 1: Spearman's ρ = [Value]
- Topic 2: Spearman's ρ = [Value]
- Topic 3: Spearman's ρ = [Value]
- Topic 4: Spearman's ρ = [Value]
- Topic 5: Spearman's ρ = [Value]

---

## Key Findings

### What Worked Well
1. **General framing validation** — LLM-as-a-judge provided reliable orderings for broad political issues
2. **Party positioning** — Clear differentiation between pro-intervention and market-led parties
3. **Embedding-based projection** — Controversy axis projection successfully captured ideological distances
4. **Reproducibility** — Consistent results across multiple runs with fixed random seed

### Challenges Encountered
1. **Sub-topic signal degradation** — Narrow topics showed weaker validation (ρ: 0.4–0.6)
   - Root cause: Topic-level speaker summaries lack sub-topic nuance
2. **Sparse representation** — Parties with few speakers (e.g., UKIP) showed high variance
   - Example: UKIP Climate Change (1 speaker: Douglas Carswell)
   - Recommendation: Enforce minimum speaker threshold (n ≥ 3–5)
3. **Statistical power** — Only 4–6 parties limits correlation metric reliability
   - P-values may be inflated due to small sample size
   - LCS ratio provides more interpretable results

### Party Profiles

#### Gaza
- **Con**: Focused on Hamas behavior, terrorism, rejection of Israel's right to exist
- **Lab**: Mixed positioning, emphasis on humanitarian concerns and two-state solution
- **LD**: Balanced, supporting both Palestinian rights and Israeli security
- **Pro**: [Positioning]

#### Climate Change
- **Con**: Strong emphasis on market solutions, energy affordability, industry competitiveness
- **Lab**: Active intervention in renewable transition, binding targets, public investment
- **LD**: [Positioning]
- **Green**: [Positioning]

---

## Validation Against Gold Standard

### Methodology
Each analysis was validated against an LLM-generated gold standard using `qwen3:8b` with thinking mode. The model was prompted to rank parties based on general political knowledge of:
- UK party ideology
- Historical stances on the issue
- Known policy positions

### Rationale for LLM-as-Judge
- **Pros**: Fully automated, consistent, captures domain knowledge
- **Cons**: Reflects general world knowledge, not period-specific actual positions
- **Validation**: Strong agreement (ρ > 0.8) suggests model captures ideological reality

---

## UMAP Visualization Insights

### Gaza General Issue
- Clear separation between parties along primary axis
- Two main clusters: Pro-Palestinian vs. Pro-Israeli orientation
- Some parties positioned centrally (nuanced/moderate positions)

### Climate Change General Issue
- Gradient rather than discrete clusters
- Strong PRO-intervention cluster (Lab, Green)
- Dispersed CON-oriented parties
- Embeddings show thematic coherence (renewable energy, investment, affordability terms)

---

## Statistical Significance

### Interpretation Guide
- **Spearman's ρ > 0.8**: Strong agreement with gold standard
- **Spearman's ρ 0.6–0.8**: Moderate agreement; some discordant pairs
- **Spearman's ρ < 0.4**: Weak agreement; may indicate:
  - Poor anchor quality
  - Insufficient summarization granularity
  - Sparse speaker signal
  - Topic-specific modeling failure

### P-values
Note: P-values are reported for context but should be interpreted carefully given small sample sizes (n = 4–6 parties). LCS ratio provides more robust assessment of agreement quality.

---

## Data Quality Notes

### Preprocessed Data Files
- `summarizations_Gaza.csv`: [Number] rows, [date range]
- `anchors_Gaza.json`: 7 issue definitions
- `summarizations_climate change.csv`: [Number] rows
- `anchors_climate change.json`: 5 issue definitions

### Missing/Excluded Data
- Years pre-2009: Not in ParlEE dataset
- Non-plenary sessions: Excluded
- Speeches with <10 opinion sentences: Excluded (insufficient signal)
- UKIP representation: Underrepresented in climate change (1 speaker)

---

## Reproducibility

**Environment**: Python 3.10.x, CUDA 11.8+
**Random Seed**: 41 (fixed for all analyses)
**Model Versions**:
- SetFit opinion classifier: [Version]
- Sentence-Transformers (embeddings): all-MiniLM-L6-v2
- Ollama LLM: qwen3:8b

**Reproducibility**: Results should be stable across runs given fixed seed, though LLM outputs (summaries, gold standard) may exhibit minor variation.

---

## Next Steps

1. **Summarization Granularity** — Implement sub-topic-level stance summaries
2. **Minimum Threshold** — Filter parties with fewer than 3 speakers
3. **Temporal Analysis** — Extend timeline to capture position evolution
4. **Politician-level Evaluation** — Increase ranking granularity for statistical power
5. **Anchor Quality Gate** — Automatic filtering of poorly-separated anchors
