| Method | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | METEOR | Kendall's Tau | Length (chars) |
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

Kendall's Tau convention: timeline-aware methods emit events in canonical timeline order and report tau = 1.000 ("by design"), conditional on the per-run strict monotonicity check of the emitted event-ID sequence (field event_order_monotonic in results_all_methods.json). The sentence-level heuristic matcher estimate — sensitive to per-event version choice, not only to ordering — is preserved as the diagnostic field tau_heuristic_matcher. The heuristic matcher remains the reported tau only for the timeline-agnostic LexRank baseline. Degradation rows: tau = 1.000 by design among the surviving events.

## Selection-level evaluation (Task 5b)

Oracle = version with highest ROUGE-L F1 vs the event's golden segment. Contested events: 96 by references, 88 evaluated (excluded: 9, 15, 28, 98, 99, 120, 131, 162; see selection_eval.json).
Random floor: analytical 0.349 (spec 96-event set), empirical 0.380 (evaluated set).

| Strategy | Oracle accuracy | ROUGE-1 F1* | ROUGE-2 F1* | ROUGE-L F1* | BERTScore F1* | METEOR* | Kendall's Tau* | Length* |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Timeline+Random | 0.372 ± 0.052 | 0.770 ± 0.018 | 0.556 ± 0.025 | 0.504 ± 0.029 | 0.912 ± 0.032 | 0.369 ± 0.017 | 1.000 ± 0.000 | 27,904 ± 1,241 |
| Timeline+Priority | 0.409 (36/88) | 0.771 | 0.581 | 0.544 | 0.874 | 0.357 | 1.000 | 27,176 |
| Timeline+Centroid | 0.375 (33/88) | 0.779 | 0.583 | 0.531 | 0.911 | 0.373 | 1.000 | 27,848 |
| Timeline+Longest | 0.648 (57/88) | 0.893 | 0.757 | 0.723 | 0.999 | 0.485 | 1.000 | 36,907 |
| TAEG (Algorithm 1) | 0.489 (43/88) | 0.829 | 0.593 | 0.530 | 0.879 | 0.426 | 1.000 | 32,199 |
| TAEG w/o BEFORE | 0.341 (30/88) | 0.767 | 0.563 | 0.512 | 0.911 | 0.365 | 1.000 | 27,235 |
| TAEG w/o SAME_EVENT | 0.489 (43/88) | 0.852 | 0.644 | 0.590 | 0.964 | 0.445 | 1.000 | 33,599 |

\* corpus metrics restricted to the evaluated contested events (hypothesis and reference sides); Kendall's Tau follows the by-design reporting convention (see note above).

### Percentile within the seeded random distribution

| Method | Oracle accuracy | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | BERTScore F1 | METEOR |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| taeg | 100% | 100% | 100% | 100% | 17% | 100% |
| longest | 100% | 100% | 100% | 100% | 100% | 100% |

Kendall's Tau is excluded from the percentile analysis: it is 1.000 by design for every timeline-aware run.
