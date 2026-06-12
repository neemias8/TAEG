
# **Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts**


This repository contains the official implementation and resources for the paper **"Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts"**, currently under revision at the **Journal of the Brazilian Computer Society (JBCS)**.

## **Abstract**

Processing overlapping narrative documents, such as legal testimonies or historical accounts, often aims not for compression but for a unified, coherent, and chronologically sound text. Standard Multi-Document Summarization (MDS), with its focus on conciseness, fails to preserve narrative flow. This paper formally defines this challenge as a new NLP task: **Narrative Consolidation**, where the central objectives are chronological integrity, completeness, and the fusion of complementary details. To demonstrate the critical role of temporal structure in this task, we introduce **Temporal Alignment Event Graph (TAEG)**, a graph structure that explicitly models chronology and event alignment. By applying a standard centrality algorithm to TAEG, our method functions as a version selection mechanism, choosing the most central representation of each event in its correct temporal position. In a study on the four Biblical Gospels, this structure-focused approach guarantees perfect temporal ordering (Kendall's Tau of 1.000) by design and dramatically improves content metrics (e.g., +357.2% in ROUGE-L F1). The success of this baseline method validates the formulation of Narrative Consolidation as a relevant task and establishes that an explicit temporal backbone is a fundamental component for its resolution.


## **The Core Problem: Summarization vs. Narrative Consolidation**

The central premise of this work is a fundamental reframing of how we process multiple, overlapping narrative documents. The goal is not to make the story shorter, but to make it **whole**.

Traditional Multi-Document Summarization (MDS) is defined by its focus on **conciseness**. However, in contexts like a criminal investigation with multiple witness testimonies or a historical analysis of overlapping accounts like the Biblical Gospels, the primary objective is to produce a single, unified, and chronologically sound narrative. The final text must eliminate redundancy while integrating crucial details from all sources into a cohesive whole.

Classic graph-based algorithms like LexRank are fundamentally mismatched for this task. By optimizing for semantic centrality, they inherently ignore the chronological flow of the narrative, resulting in a collection of salient but temporally disordered facts.

This project advocates for a paradigm shift from summarization to **Narrative Consolidation**, where coherence, completeness, and temporal integrity are prioritized over brevity.

## **Temporal Alignment Event Graph (TAEG)**

As a narrative consolidation experiment, we introduce the **Temporal Alignment Event Graph (TAEG)**, a structure that prioritizes temporal order and event alignment over simple semantic similarity.

Unlike standard methods that infer structure from textual similarity, the TAEG's construction is driven by external knowledge—a pre-defined, canonical chronology of events that serves as a structural backbone.

### **TAEG Architecture**

The TAEG is a multi-relational graph designed to separate the challenges of chronological ordering and version selection:

* **Nodes**: A distinct node is created for each *version* of a canonical event. For example, if an event is described in Matthew, Mark, and Luke, three separate nodes are created for that single event.1  
* **Edges**: The graph contains two functionally distinct types of edges:  
  1. **Temporal Edges (BEFORE)**: *Directed* edges that connect nodes representing sequential events *within the same source document*. These edges form the known chronological backbone of each narrative.  
  2. **Anchoral Edges (SAME\_EVENT)**: *Undirected* edges that interconnect all nodes (versions) that refer to the *same canonical event*, creating a cluster for each event in the timeline.1

This dual-edge architecture decouples the two primary challenges: BEFORE edges solve the sequencing problem, while SAME\_EVENT edges isolate the version selection problem.

## **Methodology & Results (JBCS revision)**

The method of the paper (Algorithm 1) applies LexRank centrality **over the TAEG graph**: TF-IDF cosine weights on `BEFORE` and `SAME_EVENT` edges, PageRank-style power iteration, and, per canonical event, selection of the version with the highest centrality within its `SAME_EVENT` cluster.

**Transparency note (relabeling).** During the paper revision, the implementation of Algorithm 1 was completed and audited. The audit found that the system previously reported here as "TAEG-LexRank" (ROUGE-L F1 0.947 etc.) selected, per event, the **longest** version rather than the centrality argmax. That configuration is preserved and now reported under its correct label, **`Timeline+Longest`**; the completed Algorithm 1 implementation (centrality-based selection over the TAEG) is reported as **`TAEG`**, with its own numbers (ROUGE-L F1 0.846 etc.). Every configuration in the tables below is reported under the label that matches what the code actually does. The revision uses this to disentangle how much of the performance comes from the **external canonical timeline** (a prior shared by no standard baseline) versus the **graph structure + centrality-based selection**, via a graded ladder of timeline-aware baselines that share the *identical* consolidation loop and differ only in the per-event selection rule:

| Strategy | Per-event selection rule |
| :---- | :---- |
| `random` | uniform among available versions (30 seeds, mean ± std) |
| `priority` | fixed source order Matthew > Mark > Luke > John |
| `centroid` | highest mean TF-IDF cosine to the other versions (local only) |
| `longest` | longest text (the pre-revision published system) |
| `taeg` | highest LexRank centrality over the full TAEG (Algorithm 1) |
| `taeg-no-before` / `taeg-no-same-event` | ablations: one edge type removed before centrality |

### Consolidated results

Produced by `python run_experiments.py --all` (full tables, including the selection-level sections, in `outputs/results_table.md` / `outputs/results_table.tex`; raw data with config and git hash in `outputs/results_all_methods.json`):

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

† Timeline-aware methods report τ = 1.000 *by design*, verified per run by a strict monotonicity check on the emitted event-ID sequence (`event_order_monotonic` in `results_all_methods.json`); the heuristic matcher estimate is preserved in the JSON as `tau_heuristic_matcher`. The heuristic matcher is the reported τ only for the timeline-agnostic LexRank. See the reporting convention section below.

How to read this (the revision's framing): **the explicit temporal backbone is the dominant factor** — even random per-event selection over the timeline reaches R-L 0.813 vs 0.206 for the timeline-agnostic LexRank. Within the timeline-aware ladder, `taeg` (Algorithm 1) sits at the 100th percentile of the 30-seed random distribution on ROUGE/METEOR and on oracle accuracy (0.489 vs the 0.380 random floor), while `longest` remains the strongest selector on this corpus (oracle accuracy 0.648) — an expected consequence of a reference that was composed favoring complete accounts. The ablations show the `BEFORE` edges carry the selection signal: removing them drops `taeg` to the random floor, while removing `SAME_EVENT` edges barely changes selection. Timeline degradation degrades content metrics roughly linearly with the removed fraction (completeness, not ordering: removed events are absent from the output by construction).

### Selection-level evaluation

Corpus-level metrics are diluted on this dataset (72/169 events have a single version, so every timeline-aware strategy emits identical text for them, and BERTScore saturates). The revision therefore adds a **selection-level evaluation**: per-event golden segments are extracted from the Golden Sample's event markers, the **oracle** for each contested event (≥2 versions) is the version with the highest ROUGE-L F1 against its golden segment, and each strategy is scored by how often it picks the oracle. See the section appended to `outputs/results_table.md` and the full report in `outputs/selection_eval.json`.

### **Conciseness vs. Consolidation Analysis**

The results for the standard LexRank baseline reflect a parameter setting of 750 sentences. To show that the timeline-aware advantage is structural and not a matter of parameter tuning, the table below shows the baseline's performance across various summary lengths (the last row is the pre-revision published system, now the `Timeline+Longest` baseline).

| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | METEOR | Kendall's Tau | Length (chars) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **LexRank Baseline** |  |  |  |  |  |  |  |
| *100 sentences* | 0.296 | 0.263 | 0.129 | 0.835 | 0.097 | 0.268 | 14,710 |
| *500 sentences* | 0.804 | 0.655 | 0.206 | 0.835 | 0.361 | 0.305 | 59,408 |
| *1000 sentences* | 0.862 | 0.728 | 0.199 | 0.835 | 0.483 | 0.320 | 100,770 |
| *1500 sentences* | 0.784 | 0.733 | 0.188 | 0.835 | 0.484 | 0.320 | 128,930 |
| **Timeline+Longest** | **0.958** | **0.938** | **0.947** | **0.995** | **0.639** | **1.000** | **79,154** |

This analysis demonstrates that simply increasing the number of sentences does not address the fundamental problem of narrative coherence: the temporal coherence (Kendall's Tau) of the timeline-agnostic baseline remains consistently low at every length. This reinforces the paper's central argument: for long and complex narratives, **an explicit temporal backbone is the dominant factor** — comprehensive coverage and chronological soundness matter far more than conciseness.

## **The Gospel Consolidation Language Resource**

To facilitate this research, we have developed and publicly released the **Gospel Consolidation Language Resource**. The dataset comprises the English New International Version (NIV, 2011\) texts of the four Gospels, mapped to 169 canonical events from the Holy Week, and a high-quality, manually created reference consolidation (the "Golden Sample").

### **Language and Version Agnostic Format**

A crucial design choice was the use of a 'book:chapter:verse' system for alignment. This decouples the chronological structure from any specific translation or language. This means other researchers can easily apply our framework to the Gospels in different languages or biblical versions (e.g., KJV, ESV) without recreating the temporal alignment from scratch.

The system is designed to parse any XML file where book, chapter, and verse identifiers are clearly tagged as attributes. As long as the verse references can be parsed, the TAEG can align them regardless of the specific XML schema or textual content.

For example, consider Matthew 21:1 from two different versions in a simple XML format:

**NIV (New International Version) XML:**

XML

\<bible version\="NIV"\>  
  \<book name\="Matthew"\>  
    \<chapter number\="21"\>  
      \<verse number\="1"\>As they approached Jerusalem and came to Bethphage on the Mount of Olives, Jesus sent two disciples,\</verse\>  
    \</chapter\>  
  \</book\>  
\</bible\>

**KJV (King James Version) XML:**

XML

\<bible version\="KJV"\>  
  \<book name\="Matthew"\>  
    \<chapter number\="21"\>  
      \<verse number\="1"\>And when they drew nigh unto Jerusalem, and were come to Bethphage, unto the mount of Olives, then sent Jesus two disciples,\</verse\>  
    \</chapter\>  
  \</book\>  
\</bible\>

Our system aligns both passages to the same canonical event by parsing the attributes book name="Matthew", chapter number="21", and verse number="1", making the framework highly adaptable and reusable across different XML-formatted biblical texts.

## **Getting Started**

### **Installation**

To set up the environment and install the required dependencies, follow these steps:

Bash

\# Clone the repository  
git clone https://github.com/neemias8/TAEG.git  
cd TAEG

\# Install dependencies  
pip install \-r requirements.txt

### **Reproducing every number of the revised paper**

One command runs the full experimental protocol — the LexRank baseline (750 sentences), all timeline-aware strategies (`random` over 30 seeds with mean ± std), both TAEG ablations, the timeline degradation experiment (10/25/50% × 10 seeds) and the selection-level evaluation:

```bash
python run_experiments.py --all
```

Generated in `outputs/`:

| File | Content |
| :---- | :---- |
| `results_all_methods.json` | every metric for every method + config, timestamp, git commit hash |
| `results_table.md` / `results_table.tex` | consolidated tables (markdown + LaTeX rows for the paper) |
| `selection_eval.json` | Task 5b report: golden segments, oracles, accuracies, percentiles |
| `results_degradation.json` | timeline degradation report (mean ± std per level) |
| `ablation_divergence.json` | per-event selection divergence of each ablation vs full `taeg` |
| `selection_report_<method>.json` | per-event candidates, scores and chosen version |
| `summary_<method>.txt` | consolidated narrative per deterministic method |

Useful options: `--methods lexrank,longest,taeg` (subset), `--random-seeds N`, `--skip-degradation`, `--output-dir DIR`.

Notes: all methods are evaluated by the same evaluator instance against the same Golden Sample; every randomized component uses fixed, logged seeds; every timeline-aware run is verified by a strict event-order monotonicity check (`event_order_monotonic` in the results JSON — see the Kendall's Tau reporting convention below); on Windows the runner forces UTF-8 output (the legacy scripts require `PYTHONUTF8=1` because of emoji prints).

### **Single runs (legacy CLI)**

```bash
# Timeline-agnostic LexRank baseline
python src/main.py --method lexrank --summary-length 750

# Any timeline-aware strategy
python src/main.py --method taeg
python src/main.py --method longest
python src/main.py --method random --seed 42

# "lexrank-ta" is kept as a backward-compatible alias for "longest"
python src/main.py --method lexrank-ta
```

### **Tests**

```bash
python -m pytest tests/
```

Covers: byte-identity of `longest` with the pre-revision published output, strategy determinism and tie-breaking, TAEG centrality (synthetic graph + convergence), ablations, timeline degradation, golden-segment parsing, the selection-level oracle, the event-order monotonicity check and the Kendall's Tau reporting convention (30 tests).

## Evaluation Metrics

### ROUGE
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest Common Subsequence

### METEOR
Word alignment-based metric with synonymy and stemming.

### BERTScore
BERT embedding-based metric for semantic similarity.

### Kendall's Tau
Ranking correlation between event order in the generated narrative and the canonical timeline. Values range from -1 (perfect disagreement) to +1 (perfect agreement). Timeline-aware methods report τ = 1.000 by design (verified per run by a monotonicity check); the heuristic sentence→event matcher is reported only for the timeline-agnostic LexRank baseline (τ = 0.320). See the reporting convention below.

## 🔧 Kendall's Tau: reporting convention

Timeline-aware methods emit events in canonical timeline order by construction. This is **verified, not assumed**: every run (including each random seed and each degradation run) passes a strict monotonicity check on the emitted event-ID sequence, recorded as `event_order_monotonic` in `outputs/results_all_methods.json`. Reporting convention:

- **Timeline-aware methods report Kendall's Tau = 1.000 ("by design")**, conditional on that run's monotonicity check having passed.
- The sentence→event heuristic matcher used previously to estimate τ is sensitive to the per-event *version choice*, not only to ordering — its values for timeline-aware methods (e.g. 0.467 for `taeg` despite perfect order) are a measurement artifact. The heuristic estimate is preserved in the JSON as the diagnostic field `tau_heuristic_matcher`, outside the main tables.
- The heuristic matcher remains the reported τ **only** for the timeline-agnostic LexRank baseline, which shows genuine partial disorder (τ = 0.320).
- Degradation rows: τ = 1.000 by design among the surviving events (removed events are absent from the output).

## Dependencies

- `beautifulsoup4`: XML/HTML processing
- `lxml`: Efficient XML parser
- `lexrank`: LEXRANK algorithm
- `nltk`: Natural language processing
- `rouge-score`: ROUGE metrics
- `bert-score`: BERTScore metric
- `transformers`: Language models
- `torch`: Deep learning framework
- `scipy`: Scientific computing
- `scikit-learn`: TF-IDF vectorization (edge weights and centroid strategy)
- `pandas`: Data manipulation
- `numpy`: Numerical computing


## Contribution

To contribute to the project:

1. Fork the repository
2. Create a branch for your feature
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is distributed under the MIT license. See the LICENSE file for more details.

## Contact

For questions or suggestions, contact the development team.