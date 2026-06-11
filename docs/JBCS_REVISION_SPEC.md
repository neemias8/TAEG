# JBCS Revision — Implementation Specification

**Context:** The paper "Narrative Consolidation: Formulating a New Task for Unifying
Multi-Perspective Accounts" received a *Revisions Required* decision at JBCS. The
reviewers' central request: disentangle how much of the TAEG's performance comes from
the **external canonical timeline** (a prior shared by no baseline so far) versus the
**graph structure + centrality-based version selection**. This spec defines exactly
what to implement. Work through the tasks **in order** — Task 1 is a prerequisite for
everything else.

**Authoritative reference for the method:** Algorithm 1 of the paper —
`ConsolidateNarrative(D, T)`: build TAEG → run LexRank centrality over the graph →
for each event in the timeline, select the version (node) with the maximum centrality
score within its `SAME_EVENT` cluster → concatenate in timeline order.

---

## Task 0 — Read and map the code (no changes)

Read: `src/main.py`, `src/summarizer.py`, `improved_graph_builder.py`,
`src/evaluator.py`, `src/data_loader.py`, `compare_methods.py`, `analyze_conciseness.py`.
Confirm the finding below and report back before coding:

> In `LexRankTemporalAnchoring.summarize_with_temporal_anchoring(use_best_gospel=True)`
> (the path used by `main.py` for `"lexrank-ta"`), per-event version selection is
> `max(event_texts, key=lambda x: len(x[1]['text']))` — i.e., **longest text**, not
> LexRank centrality over the TAEG. `ImprovedTemporalGraphBuilder` builds the graph
> (per-gospel nodes, `BEFORE` and `SAME_EVENT` edges) but no centrality is computed
> anywhere over it.

---

## Task 1 — Implement the real TAEG method (Algorithm 1 of the paper)

**Goal:** version selection by LexRank centrality computed **over the TAEG graph**.

Implementation outline (new module `src/taeg_centrality.py` or extend the summarizer):

1. Build the TAEG with `ImprovedTemporalGraphBuilder` (per-gospel version nodes,
   `BEFORE` + `SAME_EVENT` edges).
2. Assign **edge weights**:
   - `SAME_EVENT` edges: TF-IDF cosine similarity between the two versions' texts
     (reuse the vectorization approach already used by `LexRankSummarizer`).
   - `BEFORE` edges: TF-IDF cosine similarity between the connected versions' texts
     (fallback constant weight if similarity is degenerate). Treat as connections in
     the adjacency for centrality purposes.
3. Run LexRank (PageRank-style power iteration with damping ~0.85 over the weighted
   adjacency, rows normalized) to obtain a centrality score per **node** (version).
4. Consolidation loop = existing timeline iteration in
   `summarize_with_temporal_anchoring`, but selecting, per event, the version with
   the **highest centrality score** in its `SAME_EVENT` cluster (events with a single
   version: trivially selected).
5. Expose as a new selection strategy (see Task 2) named `taeg`.

Notes:
- `numpy`/`scikit-learn` are already dependencies; do not add heavy new deps.
- Deterministic output (no randomness in this path).
- Log, per event, the chosen gospel and its score (similar to the current prints),
  and also write a per-event selection report to `outputs/selection_report_taeg.json`
  (event_id, candidates, scores, chosen) — useful for the paper's qualitative analysis.

**Acceptance:** running the pipeline with the `taeg` strategy produces a consolidated
narrative; selection demonstrably uses centrality (unit check: at least some events
where the chosen version is NOT the longest one — log how many).

---

## Task 2 — Pluggable selection strategies (timeline-aware baselines)

**Goal:** refactor the per-event selection into a strategy function so all methods
share the identical timeline loop and differ ONLY in the selection criterion
(this is the fairness requirement of Reviewer G).

Strategies to implement (signature: `select(candidates: list[(node_id, node_data)]) -> (node_id, node_data)`):

| Strategy key | Selection rule |
|---|---|
| `longest`  | longest text (this is the CURRENT behavior — preserve it, relabeled) |
| `random`   | uniform random among available versions; seeded |
| `priority` | fixed source order: Matthew > Mark > Luke > John (make the order a parameter) |
| `centroid` | version with highest mean TF-IDF cosine similarity to the other versions of the SAME event (local-only selection; no global graph) |
| `taeg`     | Task 1: max LexRank centrality over the full TAEG |

Requirements:
- `random`: run N=30 seeds (configurable); evaluate each run; report mean ± std per metric.
- CLI: extend `src/main.py` (and/or a new `run_experiments.py`) to accept
  `--method {lexrank, longest, random, priority, centroid, taeg, ...}`.
- All strategies go through the SAME evaluator with the SAME Golden Sample.

**Acceptance:** `python run_experiments.py --all` evaluates every strategy and writes
`outputs/results_all_methods.json` plus a human-readable table (see Task 5).

---

## Task 3 — Ablations of the TAEG (Reviewer G, item 3)

Two variants of the `taeg` strategy (flags on the graph construction / centrality):

1. `taeg-no-before`: remove `BEFORE` edges from the graph before computing centrality;
   keep the final timeline loop intact (output order unchanged by construction).
2. `taeg-no-same-event`: remove `SAME_EVENT` edges before computing centrality;
   keep the timeline loop.

**Acceptance:** both variants run end-to-end and appear in the consolidated results
table. Report, for each ablation, how many per-event choices differ from full `taeg`
(selection-divergence count) in `outputs/ablation_divergence.json`.

---

## Task 4 — Timeline degradation experiment (optional but recommended)

For degradation levels {10%, 25%, 50%}: randomly remove that fraction of events from
the canonical timeline (`ChronologyLoader.load_chronology()` output) BEFORE building
the TAEG; run the `taeg` strategy; evaluate against the (full) Golden Sample.
N=10 seeds per level; report mean ± std. Removed events are simply absent from the
output (this measures completeness/content degradation; Kendall stays 1.0 among the
remaining events by construction — note this in the output report).

**Acceptance:** `outputs/results_degradation.json` + rows in the consolidated table.

---

## Task 5 — Unified experiment runner and paper-ready tables

Create `run_experiments.py` (repo root) that:
1. Runs: `lexrank` (750 sentences, as in the paper) + all timeline-aware strategies
   + both ablations (+ degradation if Task 4 enabled).
2. Writes `outputs/results_all_methods.json` (per method: all metrics, length in
   chars, seeds/std where applicable, timestamp, git commit hash).
3. Emits two formatted tables to `outputs/`:
   - `results_table.md` (markdown) and
   - `results_table.tex` (LaTeX rows ready to paste into the paper's Tables 4–5;
     metrics rows: ROUGE-1/2/L F1, BERTScore F1, METEOR, Kendall's Tau, Length).
4. Prints a summary to stdout.

**Acceptance:** one command reproduces every number that will appear in the revised
paper. Update `README.md`: replace the "LREC 2026" mention with JBCS (under revision),
document the new runner and strategies, and update the results table after the final
run (the current README table reports the longest-selection system as "TAEG-LexRank";
after Task 1, those numbers correspond to the `longest` baseline).

---

## Task 5b — Selection-level (discriminative) evaluation — REQUIRED

**Motivation:** corpus-level metrics will be diluted and may barely separate the
timeline-aware strategies: 72/169 events (42.6%) have a single version (identical
output for ALL strategies), parallel gospel pericopes are mutually similar, and
BERTScore saturates (~0.99 for every timeline-aware method). The discriminative
signal lives at the per-event selection level.

**Key enabler:** `data/Golden_Sample.txt` is segmented by event — the text contains
numeric markers ("1 ", "2 ", ... ) corresponding to the canonical event IDs. Build a
parser that splits the Golden Sample into per-event golden segments (validate the
split: 169 segments expected; log and handle any gaps/empty segments such as
event 162).

Implement (module `src/selection_eval.py`):

1. **Oracle selection accuracy.** For each *contested* event (>=2 versions; there are
   96: 21 with 2, 51 with 3, 24 with 4 versions), define the oracle = the version with
   the highest similarity (ROUGE-L F1 or TF-IDF cosine; report which) to that event's
   golden segment. For each strategy, report the % of contested events where it picks
   the oracle. Analytical floor for `random`: ~35% on this dataset
   ((21·1/2 + 51·1/3 + 24·1/4)/96). Report this floor alongside.
2. **Contested-subset corpus metrics.** Recompute the main metrics restricted to the
   concatenation of the 96 contested events (both hypothesis and reference sides).
3. **Random distribution as yardstick.** From the 30 `random` seeds, report the
   percentile at which `taeg` and `longest` fall within the random distribution, per
   metric and for oracle accuracy.
4. Outputs: `outputs/selection_eval.json` + rows/section in `results_table.{md,tex}`.

These analyses correspond to a planned addition to the paper's evaluation paradigm
(selection-level evaluation for extractive Narrative Consolidation), so keep the
implementation clean and well-documented.

## Important framing for result interpretation (do not skip)

- The current published "TAEG" numbers (ROUGE-1 .958 / R-2 .938 / R-L .947 /
  BERTScore .995 / METEOR .639) were produced by what is now the `longest` baseline.
  After Task 1, the paper's Table 2 will report the TRUE `taeg` numbers, and
  `longest` becomes a timeline-aware baseline in Table 4.
- It is an acceptable scientific outcome if `taeg` ties with or even loses to
  `longest` on some metrics: the paper's revised framing presents a graded ladder of
  baselines for a NEW task, and the central thesis is that the explicit temporal
  backbone is the dominant factor. Report honestly; never tune to beat a baseline.
- Keep every method's evaluation bit-for-bit comparable (same evaluator config,
  same reference, same preprocessing).

## Out of scope (deliberately — do not implement)

- Automatic timeline induction / event alignment (subject of a separate ongoing study).
- Abstractive generation (covered by a companion IJCNN 2026 paper).
- TLS/MDTS external systems as baselines (date-driven; incompatible with this corpus).

## Definition of done

1. Tasks 1–3 + 5 complete (Task 4 if time allows); all tests/runs green.
2. `outputs/results_all_methods.json` + `results_table.{md,tex}` generated.
3. README updated; per-event selection report for `taeg` available.
4. Clean commits, one logical change per commit; no changes under `data/`.
