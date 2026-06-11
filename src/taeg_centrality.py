"""
LexRank centrality over the TAEG graph — Algorithm 1 of the paper
(JBCS revision, Task 1 of docs/JBCS_REVISION_SPEC.md).

ConsolidateNarrative(D, T): build the TAEG (one node per gospel version of
each canonical event; BEFORE + SAME_EVENT edges), run LexRank centrality over
the weighted graph, then — inside the shared timeline loop — select, per
event, the version with the maximum centrality score within its SAME_EVENT
cluster.

Design decisions (documented for the paper):
- Edge weights are TF-IDF cosine similarities between the connected
  versions' texts (scikit-learn TfidfVectorizer fit over all node texts).
- Degenerate similarities (empty texts or zero cosine) fall back to a small
  constant weight so the graph stays connected along the timeline backbone.
- BEFORE edges are treated as symmetric connections in the adjacency, as in
  standard LexRank (an undirected similarity graph); the temporal direction
  is enforced by the consolidation loop itself, not by the centrality.
- Centrality = PageRank-style power iteration with damping 0.85 over the
  row-normalized weighted adjacency. Fully deterministic.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Allow importing improved_graph_builder from the repo root (same pattern as
# summarizer.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

DAMPING = 0.85
FALLBACK_EDGE_WEIGHT = 0.05  # constant for degenerate TF-IDF cosine
TOL = 1e-10
MAX_ITER = 1000


def ablation_flags(strategy_key: str) -> Dict[str, bool]:
    """Map a taeg strategy key to graph-construction flags (Task 3)."""
    flags = {
        'taeg': {},
        'taeg-no-before': {'include_before': False},
        'taeg-no-same-event': {'include_same_event': False},
    }
    if strategy_key not in flags:
        raise ValueError(f"Unknown taeg variant: {strategy_key}")
    return flags[strategy_key]


def compute_taeg_centrality(graph: Dict,
                            include_before: bool = True,
                            include_same_event: bool = True,
                            damping: float = DAMPING,
                            tol: float = TOL,
                            max_iter: int = MAX_ITER,
                            fallback_weight: float = FALLBACK_EDGE_WEIGHT
                            ) -> Tuple[Dict[str, float], Dict]:
    """
    Compute LexRank centrality per TAEG node (gospel version).

    Args:
        graph: output of ImprovedTemporalGraphBuilder.build_improved_temporal_graph
        include_before / include_same_event: ablation flags (Task 3); the
            consolidation timeline loop is unaffected by either flag
        damping, tol, max_iter: power-iteration parameters
        fallback_weight: constant weight for degenerate similarities

    Returns:
        (centrality, info) where centrality maps node_id -> score and info
        records the configuration and convergence diagnostics.
    """
    node_ids = sorted(graph['nodes'].keys())
    index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    # TF-IDF over all node texts (empty texts become zero vectors).
    texts = [graph['nodes'][nid]['text'] or "" for nid in node_ids]
    tfidf = TfidfVectorizer().fit_transform(texts)
    sim = cosine_similarity(tfidf)

    weights = np.zeros((n, n))
    used_edges = {'BEFORE': 0, 'SAME_EVENT': 0}
    fallback_edges = 0

    for edge in graph['edges']:
        etype = edge['type']
        if etype == 'BEFORE' and not include_before:
            continue
        if etype == 'SAME_EVENT' and not include_same_event:
            continue
        i, j = index[edge['from']], index[edge['to']]
        w = float(sim[i, j])
        if w <= 0.0:
            w = fallback_weight
            fallback_edges += 1
        # Symmetric adjacency (undirected similarity graph, as in LexRank).
        weights[i, j] = w
        weights[j, i] = w
        used_edges[etype] += 1

    # Row-normalize into a transition matrix; dangling nodes get a uniform row.
    row_sums = weights.sum(axis=1)
    transition = np.full((n, n), 1.0 / n)
    nonzero = row_sums > 0
    transition[nonzero] = weights[nonzero] / row_sums[nonzero, None]

    # Power iteration: p <- d * P^T p + (1 - d)/n
    p = np.full(n, 1.0 / n)
    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        p_next = damping * (transition.T @ p) + (1.0 - damping) / n
        if np.abs(p_next - p).sum() < tol:
            p = p_next
            converged = True
            break
        p = p_next

    centrality = {nid: float(p[index[nid]]) for nid in node_ids}
    info = {
        'n_nodes': n,
        'edges_used': used_edges,
        'fallback_edges': fallback_edges,
        'include_before': include_before,
        'include_same_event': include_same_event,
        'damping': damping,
        'tolerance': tol,
        'iterations': iterations,
        'converged': converged,
        'fallback_weight': fallback_weight,
    }
    return centrality, info


def build_graph_and_centrality(events: Optional[List[Dict]] = None,
                               include_before: bool = True,
                               include_same_event: bool = True,
                               verbose: bool = False
                               ) -> Tuple[Dict, Dict[str, float], Dict]:
    """Convenience wrapper: build the TAEG and compute node centrality.

    Returns (graph, centrality, info)."""
    from improved_graph_builder import ImprovedTemporalGraphBuilder
    graph = ImprovedTemporalGraphBuilder().build_improved_temporal_graph(
        events=events, verbose=verbose)
    centrality, info = compute_taeg_centrality(
        graph, include_before=include_before, include_same_event=include_same_event)
    return graph, centrality, info
