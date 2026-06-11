"""
Pluggable per-event version-selection strategies for Narrative Consolidation.

All timeline-aware methods share the identical consolidation loop
(``LexRankTemporalAnchoring.consolidate_with_strategy``) and differ ONLY in
the criterion used to pick one gospel version per canonical event — the
fairness requirement of the JBCS revision (Task 2 of
docs/JBCS_REVISION_SPEC.md).

A strategy is an object with:
    key:    short identifier used in CLIs, file names and result tables
    select(candidates) -> ((node_id, node_data), scores)
where ``candidates`` is a non-empty list of ``(node_id, node_data)`` tuples
for one event (only versions with non-empty text, in canonical gospel order)
and ``scores`` maps node_id -> criterion value (None when the criterion is
not score-based, e.g. ``random``). Ties are broken deterministically by
canonical gospel order (Matthew, Mark, Luke, John).
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Canonical source order; also the insertion order used by
# ImprovedTemporalGraphBuilder, so it doubles as the deterministic tie-break.
GOSPEL_ORDER = ('matthew', 'mark', 'luke', 'john')

Candidate = Tuple[str, Dict]


def sort_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """Sort candidates in canonical gospel order (deterministic tie-break)."""
    return sorted(candidates, key=lambda c: GOSPEL_ORDER.index(c[1]['gospel']))


class SelectionStrategy:
    """Base class: selection = argmax of scores, first-max-wins on ties."""

    key: str = None

    def scores(self, candidates: List[Candidate]) -> Dict[str, float]:
        raise NotImplementedError

    def select(self, candidates: List[Candidate]) -> Tuple[Candidate, Optional[Dict[str, float]]]:
        candidates = sort_candidates(candidates)
        scores = self.scores(candidates)
        # max() returns the first maximal element, so ties resolve to the
        # earliest gospel in canonical order — same rule as the original
        # max(key=len) implementation.
        best = max(candidates, key=lambda c: scores[c[0]])
        return best, scores


class LongestStrategy(SelectionStrategy):
    """Longest text. This is the pre-revision behavior of the published
    system (relabeled from "TAEG-LexRank"; see JBCS_REVISION_SPEC.md)."""

    key = 'longest'

    def scores(self, candidates):
        return {nid: float(len(nd['text'])) for nid, nd in candidates}


class RandomStrategy(SelectionStrategy):
    """Uniform random among available versions; seeded and logged."""

    key = 'random'

    def __init__(self, seed: int):
        self.seed = seed
        self._rng = random.Random(seed)

    def select(self, candidates):
        candidates = sort_candidates(candidates)
        chosen = candidates[self._rng.randrange(len(candidates))]
        return chosen, None

    def scores(self, candidates):  # pragma: no cover - not score-based
        return None


class PriorityStrategy(SelectionStrategy):
    """Fixed source-priority order (default Matthew > Mark > Luke > John)."""

    key = 'priority'

    def __init__(self, order: Tuple[str, ...] = GOSPEL_ORDER):
        unknown = set(order) - set(GOSPEL_ORDER)
        if unknown:
            raise ValueError(f"Unknown gospels in priority order: {unknown}")
        self.order = tuple(order)

    def scores(self, candidates):
        return {nid: float(-self.order.index(nd['gospel']))
                for nid, nd in candidates}


class CentroidStrategy(SelectionStrategy):
    """Version with the highest mean TF-IDF cosine similarity to the OTHER
    versions of the same event. Local-only: the vectorizer is fit on the
    event's candidate texts alone, with no global graph information."""

    key = 'centroid'

    def scores(self, candidates):
        if len(candidates) == 1:
            return {candidates[0][0]: 1.0}
        texts = [nd['text'] for _, nd in candidates]
        try:
            tfidf = TfidfVectorizer().fit_transform(texts)
        except ValueError:
            # Degenerate vocabulary (e.g. empty after tokenization): fall
            # back to a uniform score; canonical order breaks the tie.
            return {nid: 0.0 for nid, _ in candidates}
        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0.0)
        mean_sim = sim.sum(axis=1) / (len(candidates) - 1)
        return {nid: float(mean_sim[i]) for i, (nid, _) in enumerate(candidates)}


class TAEGStrategy(SelectionStrategy):
    """Algorithm 1 of the paper: max LexRank centrality over the full TAEG.
    Centrality scores are computed once over the whole graph (see
    src/taeg_centrality.py) and injected here."""

    def __init__(self, centrality: Dict[str, float], key: str = 'taeg'):
        self.key = key
        self.centrality = centrality

    def scores(self, candidates):
        return {nid: float(self.centrality[nid]) for nid, _ in candidates}


def get_strategy(key: str, seed: Optional[int] = None,
                 priority_order: Tuple[str, ...] = GOSPEL_ORDER,
                 centrality: Optional[Dict[str, float]] = None) -> SelectionStrategy:
    """Build a strategy by key. ``taeg*`` keys require precomputed centrality."""
    if key == 'longest':
        return LongestStrategy()
    if key == 'random':
        if seed is None:
            raise ValueError("random strategy requires a seed")
        return RandomStrategy(seed)
    if key == 'priority':
        return PriorityStrategy(priority_order)
    if key == 'centroid':
        return CentroidStrategy()
    if key.startswith('taeg'):
        if centrality is None:
            raise ValueError(f"{key} strategy requires precomputed centrality scores")
        return TAEGStrategy(centrality, key=key)
    raise ValueError(f"Unknown selection strategy: {key}")
