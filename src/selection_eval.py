"""
Selection-level (discriminative) evaluation — JBCS revision, Task 5b.

Corpus-level metrics are diluted on this dataset: 72/169 events have a
single version (identical output for every timeline-aware strategy) and
parallel pericopes are mutually similar, so BERTScore saturates. The
discriminative signal lives at the per-event selection level. This module
provides:

1. A validated parser that splits data/Golden_Sample.txt into per-event
   golden segments using its numeric event markers (169 expected; known
   data quirks — empty segment 162, missing marker 169 — are logged, never
   silently dropped).
2. Oracle selection accuracy over contested events (>=2 versions). The
   oracle for an event is the version with the highest ROUGE-L F1 against
   that event's golden segment (metric choice is recorded in the output).
3. Contested-subset corpus metrics (hypothesis and reference restricted to
   the contested events).
4. Percentile placement of deterministic strategies within the seeded
   random-selection distribution.
"""

import re
from typing import Dict, List, Optional, Tuple

from rouge_score import rouge_scorer

from selection_strategies import sort_candidates

ORACLE_METRIC = 'ROUGE-L F1'  # reported in every output that uses the oracle


def parse_golden_segments(golden_text: str, n_events: int = 169
                          ) -> Tuple[Dict[int, str], Dict]:
    """
    Split the Golden Sample into per-event segments.

    Markers are bare event numbers ("1 ", "2 ", ...) in ascending order; the
    scan is sequential (marker k+1 is only searched after marker k), which
    protects against numerals inside the narrative text.

    Returns (segments, report): segments maps event_id -> stripped text;
    report logs found/missing markers, empty segments and char coverage.
    """
    positions = {}
    pos = 0
    for k in range(1, n_events + 1):
        m = re.search(rf'(?<![\w:.]){k}(?![\w:.])', golden_text[pos:])
        if m:
            start = pos + m.start()
            positions[k] = (start, start + len(str(k)))
            pos = start + len(str(k))

    segments = {}
    found = sorted(positions)
    for i, k in enumerate(found):
        seg_start = positions[k][1]
        seg_end = positions[found[i + 1]][0] if i + 1 < len(found) else len(golden_text)
        segments[k] = golden_text[seg_start:seg_end].strip()

    missing = [k for k in range(1, n_events + 1) if k not in positions]
    empty = [k for k in found if not segments[k]]
    report = {
        'expected_segments': n_events,
        'markers_found': len(positions),
        'missing_markers': missing,
        'empty_segments': empty,
        'covered_chars': sum(len(s) for s in segments.values()),
        'golden_chars': len(golden_text),
    }
    return segments, report


def percentile_in_distribution(value: float, distribution: List[float]) -> float:
    """Empirical percentile: share of the distribution <= value, in percent."""
    if not distribution:
        raise ValueError("empty distribution")
    return 100.0 * sum(1 for v in distribution if v <= value) / len(distribution)


class SelectionEvaluator:
    """Oracle and contested-subset machinery shared by all strategies."""

    def __init__(self, graph: Dict, golden_text: str, n_events: int = 169):
        self.graph = graph
        self.segments, self.parse_report = parse_golden_segments(golden_text, n_events)
        self._rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Group versions by event (contested = >=2 versions by chronology refs).
        by_event: Dict[int, list] = {}
        for nid, nd in graph['nodes'].items():
            by_event.setdefault(nd['event_id'], []).append((nid, nd))
        self._by_event = {eid: sort_candidates(nodes) for eid, nodes in by_event.items()}
        self.contested_ids = sorted(eid for eid, nodes in by_event.items()
                                    if len(nodes) >= 2)

        # Analytical floor over the spec's contested set: expected accuracy of
        # uniform random selection = mean over events of 1/n_versions.
        self.analytical_floor = sum(
            1.0 / len(self._by_event[eid]) for eid in self.contested_ids
        ) / len(self.contested_ids)

        self._build_oracles()

    def _build_oracles(self):
        """Oracle per contested event = max-ROUGE-L version vs golden segment.

        Events where selection is trivial or undefined are excluded and
        logged: <2 non-empty candidate texts (all strategies coincide or the
        choice is vacuous) or a missing/empty golden segment."""
        self.oracles: Dict[int, Dict] = {}
        self.excluded: Dict[int, str] = {}
        for eid in self.contested_ids:
            nodes = self._by_event[eid]
            nonempty = [(nid, nd) for nid, nd in nodes if nd['text']]
            segment = self.segments.get(eid, '')
            if len(nonempty) < 2:
                self.excluded[eid] = 'fewer than 2 non-empty candidate texts'
                continue
            if not segment:
                self.excluded[eid] = 'missing or empty golden segment'
                continue
            scores = {nid: self._rouge.score(segment, nd['text'])['rougeL'].fmeasure
                      for nid, nd in nonempty}
            oracle_node = max(nonempty, key=lambda c: scores[c[0]])[0]
            self.oracles[eid] = {
                'oracle_node': oracle_node,
                'oracle_gospel': self.graph['nodes'][oracle_node]['gospel'],
                'scores': scores,
                'n_versions': len(nodes),
                'n_nonempty': len(nonempty),
            }
        # Floor restricted to the actually-evaluated subset.
        if self.oracles:
            self.empirical_floor = sum(
                1.0 / o['n_nonempty'] for o in self.oracles.values()
            ) / len(self.oracles)
        else:
            self.empirical_floor = 0.0

    # ---------------- per-strategy measures ----------------

    def oracle_accuracy(self, records: List[Dict]) -> Dict:
        """% of evaluated contested events where the strategy picks the oracle."""
        chosen = {r['event_id']: r['chosen_node'] for r in records if not r['fallback']}
        hits = [eid for eid, o in self.oracles.items()
                if chosen.get(eid) == o['oracle_node']]
        return {
            'oracle_metric': ORACLE_METRIC,
            'n_evaluated': len(self.oracles),
            'n_hits': len(hits),
            'accuracy': len(hits) / len(self.oracles) if self.oracles else 0.0,
        }

    def contested_subset_texts(self, records: List[Dict]) -> Tuple[str, str]:
        """(hypothesis, reference) restricted to the evaluated contested
        events, both sides concatenated in timeline order."""
        chosen = {r['event_id']: r['chosen_node'] for r in records if not r['fallback']}
        hyp_parts, ref_parts = [], []
        for eid in sorted(self.oracles):
            node = chosen.get(eid)
            hyp_parts.append(self.graph['nodes'][node]['text'] if node else '')
            ref_parts.append(self.segments[eid])
        return ' '.join(hyp_parts), ' '.join(ref_parts)

    def summary(self) -> Dict:
        """Static description of the oracle setup (for the JSON report)."""
        return {
            'oracle_metric': ORACLE_METRIC,
            'golden_parse': self.parse_report,
            'n_contested_by_references': len(self.contested_ids),
            'n_evaluated': len(self.oracles),
            'excluded_events': {str(k): v for k, v in sorted(self.excluded.items())},
            'analytical_random_floor_spec_set': self.analytical_floor,
            'empirical_random_floor_evaluated_set': self.empirical_floor,
            'oracles': {
                str(eid): {
                    'oracle_node': o['oracle_node'],
                    'oracle_gospel': o['oracle_gospel'],
                    'n_versions': o['n_versions'],
                    'n_nonempty': o['n_nonempty'],
                    'scores': o['scores'],
                }
                for eid, o in sorted(self.oracles.items())
            },
        }
