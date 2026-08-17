
# **Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts**


This repository contains the official implementation and resources for the paper **"Narrative Consolidation: Formulating a New Task for Unifying Multi-Perspective Accounts"**, **accepted for publication** at the **Journal of the Brazilian Computer Society (JBCS)**.

## **Abstract**

Processing overlapping narrative documents, such as legal testimonies or historical accounts, often aims not for compression but for a unified, coherent, and chronologically sound text. Standard Multi-Document Summarization (MDS), with its focus on conciseness, fails to preserve narrative flow. This paper formally defines this challenge as a new NLP task, **Narrative Consolidation**, whose central objectives are chronological integrity, completeness, and the fusion of complementary details, and establishes the resources needed to study it: a formal task definition, an evaluation paradigm that includes a selection-level metric, the **Gospel Consolidation Language Resource** — a benchmark built from the four Biblical Gospels, with 169 canonical events, cross-document alignments, and a manually created reference consolidation — and a suite of reference systems, ranging from a timeline-agnostic centrality method to timeline-aware heuristics and the **Temporal Alignment Event Graph (TAEG)**, a multi-relational graph that explicitly models chronology and event alignment.

Benchmarking these systems yields three findings that characterize the task. First, **the explicit temporal backbone is the dominant factor**: every system granted the canonical timeline raises ROUGE-L F1 from 0.206 to at least 0.81, whereas content-selection sophistication accounts for a far smaller margin. Second, on a fusion-style reference, **a simple length heuristic — selecting the longest available account of each event — is remarkably strong** (0.947 ROUGE-L F1; 0.648 selection accuracy), outperforming the graph-based selection (0.846; 0.489), which is nevertheless clearly above the random floor. Third, ablations show that **the discriminative signal resides in the temporal edges, while intra-cluster lexical similarity is uninformative** — a negative result that constrains the design of future models. Together, these results establish Narrative Consolidation as a task distinct from summarization, provide the first reference points for it, and pose an explicit open challenge: to surpass that heuristic with a principled selection mechanism, and to move beyond version selection towards the fusion of complementary details.


## **The Core Problem: Summarization vs. Narrative Consolidation**

The central premise of this work is a fundamental reframing of how we process multiple, overlapping narrative documents. The goal is not to make the story shorter, but to make it **whole**.

Traditional Multi-Document Summarization (MDS) is defined by its focus on **conciseness**. However, in contexts like a criminal investigation with multiple witness testimonies or a historical analysis of overlapping accounts like the Biblical Gospels, the primary objective is to produce a single, unified, and chronologically sound narrative. The final text must eliminate redundancy while integrating crucial details from all sources into a cohesive whole.

Classic graph-based algorithms like LexRank are fundamentally mismatched for this task. By optimizing for semantic centrality, they inherently ignore the chronological flow of the narrative, resulting in a collection of salient but temporally disordered facts.

This project advocates for a paradigm shift from summarization to **Narrative Consolidation**, where coherence, completeness, and temporal integrity are prioritized over brevity.

## **Reference Systems for Narrative Consolidation**

A new task requires reference points against which future work can be measured. This repository implements a graded ladder of reference systems, organized along the two factors that plausibly drive performance on the task: access to a canonical timeline, and the sophistication of the per-event version-selection criterion. **None of these systems is advanced as *the* solution to Narrative Consolidation** — their role is to delimit what is easy, what is hard, and what remains open.

### **Timeline-agnostic reference: LexRank**

At the bottom of the ladder sits the standard approach for graph-based extractive summarization, treating the four Gospels as a single multi-document collection: sentences are nodes, edge weights are TF-IDF cosine similarity, and LexRank ranks and selects the top-scoring sentences. This baseline inherently ignores chronological flow (see the *Conciseness vs. Consolidation* analysis below).

### **Timeline-aware heuristic baselines**

Above it sit four heuristics that receive *exactly the same* canonical timeline (the 169-event Holy Week chronology) and iterate over it, differing only in the per-event selection rule: `Timeline+Random` (uniform, establishing the performance floor), `Timeline+Priority` (fixed source order Matthew ≻ Mark ≻ Luke ≻ John), `Timeline+Centroid` (highest mean TF-IDF similarity to the event's other versions — local only), and `Timeline+Longest` (the longest available version). See the results tables below for how they compare.

### **The Temporal Alignment Event Graph (TAEG): a structured, diagnostic reference system**

Finally, the **Temporal Alignment Event Graph (TAEG)** combines the timeline with an explicit relational structure over event versions. **Its purpose in this study is diagnostic**: it lets us test whether a structured prior over event versions improves selection beyond what the timeline alone provides — not to serve as the proposed solution to the task. As reported below, the answer is a qualified "yes, but less than a length heuristic."

Unlike the heuristics above, which only consult a single event's candidate versions, the TAEG's construction is driven by external knowledge — the same pre-defined, canonical chronology — but represented as an explicit multi-relational graph over which centrality is computed.

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

How to read this: the spread among timeline-aware systems (ROUGE-L F1 0.811–0.947) is an order of magnitude smaller than the gap separating them from the timeline-agnostic LexRank (0.206) — once the chronological structure is given, most of the task is already solved, and the choice of version accounts for the remainder. Within that remainder, `taeg` (Algorithm 1) sits at the 100th percentile of the 30-seed random distribution on ROUGE/METEOR and on oracle accuracy (0.489 vs. the 0.380 random floor) — the graph carries genuine signal — but `longest` remains the strongest selector on this corpus (oracle accuracy 0.648). See **Findings** below for the full picture and its implications.

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

## **Findings**

Benchmarking the reference systems above yields four findings that characterize Narrative Consolidation as a task, not just a ranking of methods (full discussion in Section 8 of the paper):

1. **Chronological structure, not content selection, is the defining difficulty.** Granting any system the canonical timeline moves ROUGE-L F1 from 0.206 to at least 0.811 — a gap an order of magnitude larger than the spread among timeline-aware selection criteria (0.811–0.947). This is the empirical justification for treating Narrative Consolidation as a task distinct from Multi-Document Summarization.
2. **A length heuristic is a strong reference point on fusion-style references.** `Timeline+Longest` reaches 0.947 ROUGE-L F1 and 0.648 oracle selection accuracy, ahead of every other system, because the Golden Sample fuses complementary, non-conflicting sources and the most detailed account tends to overlap it most. Any future system evaluated on this resource should be compared against it, not only against a random or centrality-based baseline.
3. **An explicit relational prior yields signal, but not enough.** The TAEG's centrality-based selection (oracle accuracy 0.489) sits well above the random floor (≈0.37) and at the 100th percentile of a 30-seed random distribution — the graph carries genuine information — yet remains behind `Timeline+Longest`. Surpassing the length heuristic with a principled selection mechanism, on this benchmark, is the concrete open challenge the resource poses.
4. **Intra-event lexical similarity is uninformative for version selection.** Ablating the `SAME_EVENT` edges leaves oracle accuracy unchanged (0.489 → 0.489) and even slightly improves corpus metrics; ablating `BEFORE` edges collapses it to the random floor (0.341). The discriminative signal resides entirely in the temporal relations — parallel accounts of the same event are nearly equidistant in TF-IDF space, so future models built on this structure (including GNNs) will need intra-event edge weights derived from semantic, discourse, or factuality features, not surface overlap.

We report Findings 2–4 candidly even though they are unfavorable to the TAEG: a benchmark that documents where simple methods are hard to beat is, in our view, more useful to the community than one reporting only favorable comparisons. **The TAEG is offered as one reference point among several, not as the solution to Narrative Consolidation.**

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

## **Companion Studies**

This task formulation is the basis of an ongoing research program, not a one-off experiment:

- **Abstractive Narrative Consolidation** — grounding a GNN encoder and an LLM decoder on the TAEG to *fuse* (not just select) the versions within each `SAME_EVENT` cluster, addressing the Representativeness objective in its strongest form. Accepted at IJCNN 2026 (Finger et al., 2026).
- **Automatic timeline induction** — removing the assumption of a known canonical timeline, which Section 8/9 of the paper identifies as the decisive open problem for the task (cross-document event extraction, coreference, and temporal ordering). Currently in progress.

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