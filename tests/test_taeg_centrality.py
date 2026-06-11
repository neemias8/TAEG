"""
Tests for LexRank centrality over the TAEG (JBCS revision, Task 1).
"""

import sys
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from taeg_centrality import ablation_flags, compute_taeg_centrality  # noqa: E402
from selection_strategies import LongestStrategy, TAEGStrategy  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def repo_cwd():
    old = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def graph():
    from improved_graph_builder import ImprovedTemporalGraphBuilder
    return ImprovedTemporalGraphBuilder().build_improved_temporal_graph(verbose=False)


def _mini_graph():
    """Three versions of one event: A and B near-duplicates, C unrelated.
    SAME_EVENT edges connect all pairs; LexRank must rank A and B above C."""
    nodes = {
        'event_1_matthew': {'event_id': 1, 'gospel': 'matthew', 'reference': '1:1',
                            'description': 'd', 'text': 'jesus entered jerusalem riding a donkey'},
        'event_1_mark': {'event_id': 1, 'gospel': 'mark', 'reference': '1:1',
                         'description': 'd', 'text': 'jesus entered jerusalem riding a young donkey'},
        'event_1_luke': {'event_id': 1, 'gospel': 'luke', 'reference': '1:1',
                         'description': 'd', 'text': 'completely different words about something else entirely'},
    }
    edges = [
        {'from': 'event_1_matthew', 'to': 'event_1_mark', 'type': 'SAME_EVENT'},
        {'from': 'event_1_matthew', 'to': 'event_1_luke', 'type': 'SAME_EVENT'},
        {'from': 'event_1_mark', 'to': 'event_1_luke', 'type': 'SAME_EVENT'},
    ]
    return {'nodes': nodes, 'edges': edges}


def test_mini_graph_ranks_mutually_similar_versions_higher():
    centrality, info = compute_taeg_centrality(_mini_graph())
    assert info['converged']
    assert centrality['event_1_matthew'] > centrality['event_1_luke']
    assert centrality['event_1_mark'] > centrality['event_1_luke']


def test_centrality_is_deterministic_and_normalized(graph):
    c1, i1 = compute_taeg_centrality(graph)
    c2, _ = compute_taeg_centrality(graph)
    assert c1 == c2
    assert i1['converged']
    assert i1['n_nodes'] == 363
    assert i1['edges_used'] == {'BEFORE': 799, 'SAME_EVENT': 318}
    total = sum(c1.values())
    assert abs(total - 1.0) < 1e-6
    assert all(v > 0 for v in c1.values())


def test_taeg_selection_differs_from_longest_on_some_events(graph):
    """Acceptance check of Task 1: selection demonstrably uses centrality —
    at least some contested events choose a version that is NOT the longest."""
    from summarizer import LexRankTemporalAnchoring
    centrality, _ = compute_taeg_centrality(graph)
    ta = LexRankTemporalAnchoring()
    _, taeg_records = ta.consolidate_with_strategy(
        TAEGStrategy(centrality), graph=graph, verbose=False)
    _, longest_records = ta.consolidate_with_strategy(
        LongestStrategy(), graph=graph, verbose=False)

    diffs = [
        (t['event_id'], t['chosen_gospel'], l['chosen_gospel'])
        for t, l in zip(taeg_records, longest_records)
        if not t['fallback'] and t['chosen_node'] != l['chosen_node']
    ]
    assert len(diffs) > 0, "taeg never deviates from longest — centrality unused?"

    # And the taeg choice must be the centrality argmax of its cluster.
    for rec in taeg_records:
        if rec['fallback']:
            continue
        best = max(rec['candidates'], key=lambda c: c['score'])
        assert rec['chosen_node'] == best['node_id']


def test_ablation_flags_change_the_scores(graph):
    full, _ = compute_taeg_centrality(graph)
    no_before, i_nb = compute_taeg_centrality(graph, include_before=False)
    no_same, i_ns = compute_taeg_centrality(graph, include_same_event=False)
    assert i_nb['edges_used']['BEFORE'] == 0
    assert i_ns['edges_used']['SAME_EVENT'] == 0
    assert full != no_before
    assert full != no_same


def test_ablation_flags_mapping():
    assert ablation_flags('taeg') == {}
    assert ablation_flags('taeg-no-before') == {'include_before': False}
    assert ablation_flags('taeg-no-same-event') == {'include_same_event': False}
    with pytest.raises(ValueError):
        ablation_flags('taeg-bogus')
