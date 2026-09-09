# Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts

[![DOI](https://img.shields.io/badge/DOI-10.5753/jbcs.2026.7717-blue)](https://doi.org/10.5753/jbcs.2026.7717)

This repository contains the official implementation and resources for the paper **"Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts"**. The project defines and tests automated methods for unifying overlapping narrative documents (such as the four Gospels' Holy Week accounts) into a single, chronologically coherent narrative guided by a canonical timeline.

## Publication

**Finger, R. A., Cortes, E. G., Rigo, S. J., & Ramos, G. de O.** (2026). Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts. *Journal of the Brazilian Computer Society*, 32(1), 2203–2215. https://doi.org/10.5753/jbcs.2026.7717

For citation in BibTeX, see `CITATION.bib`.

## Abstract

Processing overlapping narrative documents, such as legal testimonies or historical accounts, often aims not for compression but for a unified, coherent, and chronologically sound text. Standard multi-document summarization (MDS) frameworks measure success via ROUGE, prioritizing brevity. Narrative Consolidation flips this priority: the goal is not conciseness but integrity—preserving all events in their correct temporal order.

This work formalizes Narrative Consolidation as a task, proposes the Gospel Consolidation dataset (four Gospels' Holy Week accounts aligned to a canonical timeline of 169 events), and benchmarks five reference systems including LexRank (timeline-agnostic), timeline-aware heuristics (random, priority, longest, centroid), and the Temporal Alignment Event Graph (TAEG), which computes LexRank-style centrality over a relational graph of event versions.

Benchmarking systems on the Gospel Consolidation dataset yields three key findings: first, the explicit temporal backbone (a canonical timeline) is the dominant factor — granting any system the canonical timeline moves ROUGE-L F1 from 0.206 to at least 0.811. Second, a simple length heuristic is a strong reference (0.947 F1). Third, an explicit relational prior (TAEG) yields modest additional signal but does not fully close the gap, suggesting that version selection in narrative consolidation remains an open challenge.

## The Core Problem: Summarization vs. Narrative Consolidation

The central premise of this work is a fundamental reframing of how we process multiple, overlapping narrative documents. The goal is not to make the story shorter, but to make it whole.

Traditional MDS focuses on conciseness. In contexts like criminal investigations with multiple witness testimonies or historical analysis with multiple sources, the priority is often chronological coherence and completeness—ensuring all material is included and temporally sound—rather than creating the smallest possible summary.

Narrative Consolidation prioritizes coherence, completeness, and temporal integrity over brevity.

## Reference Systems for Narrative Consolidation

We implement a graded ladder of reference systems, organized along two factors that drive performance: whether the system receives the canonical timeline and the per-event version selection rule.

### Timeline-agnostic reference: LexRank

A standard graph-based extractive summarization approach that treats the four Gospels as a single multi-document collection: sentences are nodes and edges are weighted by sentence similarity. This baseline receives no canonical timeline and serves as a reference point for the inherent difficulty of synthesizing narrative from overlapping sources without temporal guidance.

### Timeline-aware heuristic baselines

These heuristics receive the canonical timeline (169-event Holy Week chronology) and iterate over it. They differ only in the per-event selection rule:
- **random**: uniform among available versions (mean ± std reported across 30 seeds)
- **priority**: fixed source order Matthew > Mark > Luke > John
- **centroid**: highest mean TF–IDF cosine to the other versions (local only)
- **longest**: longest text (this is the pre-revision published system)

A legacy alias `lexrank-ta` is kept for backward compatibility and maps to `longest`.

### The Temporal Alignment Event Graph (TAEG)

The TAEG is a structured, diagnostic reference system that combines the canonical timeline with an explicit relational graph over event versions. Its purpose is diagnostic: it lets us test whether a relational prior improves over simpler heuristics.

**TAEG Architecture:**
- **Nodes**: one node per version of each canonical event. If Matthew, Mark, and Luke all describe event E, the TAEG contains three distinct nodes for E.
- **Edges**: two main edge types are used:
  - **Temporal edges (BEFORE)**: directed edges that connect nodes representing sequential events within the same source document; they encode the known chronological backbone of each narrative.
  - **Anchoral edges (SAME_EVENT)**: undirected edges that connect all nodes referring to the same canonical event, creating a cluster per event.

This dual-edge architecture decouples sequencing (BEFORE edges) from version selection (SAME_EVENT edges).

## Methodology & Results (JBCS revision)

Algorithm 1 in the paper prescribes computing LexRank-style centrality over the full TAEG (using TF–IDF cosine weights on BEFORE and SAME_EVENT edges, PageRank-style power iteration) and then selecting, per canonical event, the version node with the highest centrality score.

Transparency note: the system previously reported in earlier pre-revision artifacts as "TAEG-LexRank" had a mismatch: the code built the graph but selected per-event versions by text length (the longest) rather than centrality. This is now corrected: the graph is built and centrality is computed; the old behavior is preserved and relabeled as the `Timeline+Longest` baseline, and the true Algorithm 1 is now the `taeg` method. All experimental runs are deterministic and logged; git commit hashes are recorded in the output JSON.

**Per-event selection strategies implemented and reported:**

| Strategy | Per-event selection rule |
| :---- | :---- |
| `random` | uniform among available versions (30 seeds, mean ± std) |
| `priority` | fixed source order Matthew > Mark > Luke > John |
| `centroid` | highest mean TF–IDF cosine to the other versions (local only) |
| `longest` | longest text (the pre-revision published system) |
| `taeg` | highest LexRank centrality over the full TAEG (Algorithm 1) |
| `taeg-no-before` / `taeg-no-same-event` | ablations: one edge type removed before centrality |

### Consolidated results

Produced by `python run_experiments.py --all`. Full tables, selection-level sections, and raw data (config + git hash) appear in `outputs/`.

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | METEOR | Kendall's Tau† | Length (chars) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| LexRank (750 sent.) | 0.887 | 0.712 | 0.206 | 0.835 | 0.453 | 0.320 | 81,418 |
| Timeline+Random (N=30) | 0.889 ± 0.009 | 0.816 ± 0.013 | 0.813 ± 0.015 | 0.932 ± 0.033 | 0.478 ± 0.026 | 1.000 ± 0.000 | 70,052 ± 1,267 |
| Timeline+Priority | 0.886 | 0.814 | 0.811 | 0.897 | 0.453 | 1.000 | 69,295 |
| Timeline+Centroid | 0.891 | 0.821 | 0.818 | 0.935 | 0.472 | 1.000 | 69,929 |
| Timeline+Longest | 0.958 | 0.938 | 0.947 | 0.995 | 0.639 | 1.000 | 79,154 |
| TAEG (Algorithm 1) | 0.918 | 0.848 | 0.846 | 0.906 | 0.550 | 1.000 | 74,280 |
| TAEG w/o BEFORE | 0.886 | 0.811 | 0.809 | 0.935 | 0.469 | 1.000 | 69,316 |
| TAEG w/o SAME_EVENT | 0.929 | 0.865 | 0.866 | 0.964 | 0.575 | 1.000 | 75,846 |
| TAEG, timeline -10% (N=10) | 0.864 ± 0.020 | 0.800 ± 0.021 | 0.795 ± 0.022 | 0.897 ± 0.015 | 0.474 ± 0.021 | 1.000 ± 0.000 | 66,198 ± 2,570 |
| TAEG, timeline -25% (N=10) | 0.783 ± 0.034 | 0.730 ± 0.035 | 0.723 ± 0.037 | 0.882 ± 0.021 | 0.397 ± 0.027 | 1.000 ± 0.000 | 55,572 ± 3,716 |
| TAEG, timeline -50% (N=10) | 0.581 ± 0.035 | 0.541 ± 0.038 | 0.534 ± 0.040 | 0.878 ± 0.033 | 0.246 ± 0.026 | 1.000 ± 0.000 | 35,193 ± 2,835 |

† Timeline-aware methods report τ = 1.000 by design; every run is verified by a strict monotonicity check on the emitted event-ID sequence (`event_order_monotonic` in `results_all_methods.json`).

### Selection-level evaluation

Corpus-level metrics are diluted on this dataset because 72/169 events have a single version; for those events all timeline-aware strategies emit identical text. Selection-level evaluation (oracle accuracy, percentile ranks) is reported separately in `selection_eval.json` and the paper.

### Conciseness vs. Consolidation Analysis

The LexRank baseline table shows multiple sentence-budget settings to demonstrate that increasing summary length does not solve narrative coherence. Timeline-aware strategies consistently achieve τ = 1.000 (perfect temporal ordering) by construction, whereas LexRank without timeline guidance scores 0.320, confirming that temporal structure is the binding constraint.

## Findings

Benchmarking the reference systems yields four main findings (see the paper for full discussion):

1. **Chronological structure, not content selection, is the defining difficulty.** Granting any system the canonical timeline moves ROUGE-L F1 from 0.206 to at least 0.811.
2. **A length heuristic is a strong reference point on fusion-style references:** `Timeline+Longest` reaches very high ROUGE-L and oracle selection accuracy on the chosen references.
3. **An explicit relational prior yields signal, but not enough:** the TAEG's centrality-based selection sits above random but below the length heuristic on these references.
4. **Intra-event lexical similarity is uninformative for version selection:** ablating SAME_EVENT edges does not meaningfully change oracle accuracy.

## The Gospel Consolidation Language Resource

We release the Gospel Consolidation Language Resource used in this work. It contains the English New International Version (NIV, 2011) texts of the four Gospels aligned using a book:chapter:verse schema and mapped to a canonical timeline of 169 Holy Week events.

The system parses any XML where book, chapter and verse identifiers are clearly tagged as attributes. This decouples the chronological structure from translation or language and makes the resource reusable for other Bible translations or narrative-fusion datasets with similar structure.

## Companion Studies

Ongoing and planned follow-ups include:
- **Abstractive Narrative Consolidation:** grounding a GNN encoder and an LLM decoder on the TAEG to fuse versions within each SAME_EVENT cluster.
- **Automatic timeline induction:** removing the assumption of a known canonical timeline.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/neemias8/TAEG.git
cd TAEG

# (recommended) create and activate a Python 3.13 venv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Reproducing the revised paper's experiments

One command runs the full experimental protocol (LexRank baseline, all timeline-aware strategies, TAEG ablations, timeline degradation runs):

```bash
python run_experiments.py --all
```

Generated in `outputs/`:

| File | Content |
| :---- | :---- |
| `results_all_methods.json` | every metric for every method + config, timestamp, git commit hash |
| `results_table.md` / `results_table.tex` | consolidated tables for the paper |
| `selection_eval.json` | golden segments, oracles, accuracies, percentiles |
| `results_degradation.json` | timeline degradation report |
| `ablation_divergence.json` | per-event selection divergence of each ablation vs full `taeg` |
| `selection_report_<method>.json` | per-event candidates, scores and chosen version |
| `summary_<method>.txt` | consolidated narrative per deterministic method |

Useful options: `--methods lexrank,longest,taeg`, `--random-seeds N`, `--skip-degradation`, `--output-dir DIR`.

**Notes:** all methods are evaluated by the same evaluator instance against the same Golden Sample; every randomized component uses fixed, logged seeds; every timeline-aware run is verified by the strict monotonicity check before results are recorded.

### Single runs (CLI)

```bash
# Timeline-agnostic LexRank baseline
python src/main.py --method lexrank --summary-length 750

# Any timeline-aware strategy
python src/main.py --method taeg
python src/main.py --method longest
python src/main.py --method random --seed 42

# Backward-compatible alias
python src/main.py --method lexrank-ta
```

### Tests

```bash
python -m pytest tests/
```

Covers: determinism of strategies, tie-breaking, TAEG centrality (synthetic graphs + convergence), ablations, timeline degradation, gold-sample identity checks.

## Evaluation Metrics

- **ROUGE-1:** Unigram overlap
- **ROUGE-2:** Bigram overlap
- **ROUGE-L:** Longest Common Subsequence
- **METEOR:** word-alignment with stemming and synonymy
- **BERTScore:** embedding-based semantic similarity
- **Kendall's Tau:** ranking correlation between emitted event order and the canonical timeline (−1 to +1). Timeline-aware methods report τ = 1.000 by design when the monotonicity check passes.

## Kendall's Tau: reporting convention

Timeline-aware methods emit events in canonical timeline order by construction; we verify this property per run. Timeline-aware rows report τ = 1.000 when the strict monotonicity check has passed. The LexRank baseline, which has no temporal constraints, scores 0.320 (indicating substantial reordering relative to the canonical sequence).

## Dependencies

See `requirements.txt` / `pyproject.toml`. Key dependencies used in evaluation and graph construction include:
- beautifulsoup4
- lxml
- lexrank
- nltk
- rouge-score
- bert-score
- transformers
- torch
- scipy
- scikit-learn
- pandas
- numpy

## Contribution

To contribute:
1. Fork the repository
2. Create a branch for your feature
3. Implement changes and add tests if applicable
4. Submit a pull request referencing the relevant items in `docs/JBCS_REVISION_SPEC.md`

## License

This project is distributed under the MIT license. See the LICENSE file for details.

## Contact

For questions or suggestions, contact the development team or the repository owner.
