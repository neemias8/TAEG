# TAEG — Narrative Consolidation

## What this project is
Official implementation of the paper "Narrative Consolidation: Formulating a New Task
for Unifying Multi-Perspective Accounts" (under revision at the Journal of the
Brazilian Computer Society — JBCS). The task: unify the four Gospels' Holy Week
accounts into a single, chronologically coherent narrative, guided by a canonical
timeline of 169 events.

## Current mission (JBCS revision)
We are implementing the experiments requested by the reviewers. The full
specification, context and acceptance criteria are in `docs/JBCS_REVISION_SPEC.md`.
**Read that file before doing any work.**

## Architecture map
- `src/main.py` — `TAEGPipeline`: orchestrates load → summarize → evaluate.
  Methods: `"lexrank"` (timeline-agnostic baseline) and `"lexrank-ta"` (timeline-aware).
- `src/summarizer.py` — `LexRankSummarizer` (baseline) and
  `LexRankTemporalAnchoring.summarize_with_temporal_anchoring()` (timeline loop;
  per-event version selection currently = longest text when `use_best_gospel=True`).
- `improved_graph_builder.py` — `ImprovedTemporalGraphBuilder`: builds the TAEG
  (one node per gospel-version of each event; `BEFORE` and `SAME_EVENT` edges).
- `src/evaluator.py` — `SummarizationEvaluator`: ROUGE, METEOR, BERTScore, Kendall's Tau.
- `src/data_loader.py` — `BiblicalDataLoader` (gospel XMLs, Golden Sample) and
  `ChronologyLoader` (169-event canonical timeline).
- `data/` — 4 gospel XMLs (NIV), `ChronologyOfTheFourGospels_PW.xml`, `Golden_Sample.txt`.
- `outputs/` — generated summaries and evaluation JSONs.

## Critical known issue (must fix first)
The paper's Algorithm 1 says version selection = LexRank **centrality over the TAEG
graph** (`RunLexRank(G)` then `FindMaxScore` per `SAME_EVENT` cluster). The code,
however, selects by **text length** (`max(event_texts, key=lambda x: len(x[1]['text']))`
in `summarize_with_temporal_anchoring`). The graph is built but centrality is never
computed. Task 1 of the spec implements the real Algorithm 1; the current behavior
is preserved and relabeled as the `Timeline+Longest` baseline.

## Conventions
- Python 3.13; deps in `requirements.txt` / `pyproject.toml`.
- Keep evaluation strictly comparable: same evaluator, same Golden Sample, for all methods.
- Deterministic where possible; random baselines use fixed, logged seeds.
- New experiment outputs go to `outputs/` as JSON + a consolidated results table
  (markdown + LaTeX) for the paper.
- Do not change `data/` files.
