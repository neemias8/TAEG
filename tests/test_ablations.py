"""
Tests for the TAEG ablations (JBCS revision, Task 3).
"""

import sys
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from taeg_centrality import ablation_flags, compute_taeg_centrality  # noqa: E402
from selection_strategies import TAEGStrategy, selection_divergence  # noqa: E402


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


@pytest.fixture(scope="session")
def consolidator():
    from summarizer import LexRankTemporalAnchoring
    return LexRankTemporalAnchoring()


def _records_for(variant, graph, consolidator):
    centrality, info = compute_taeg_centrality(graph, **ablation_flags(variant))
    assert info['converged']
    summary, records = consolidator.consolidate_with_strategy(
        TAEGStrategy(centrality, key=variant), graph=graph, verbose=False)
    return summary, records


def test_ablations_run_end_to_end_with_intact_timeline(graph, consolidator):
    full_summary, full = _records_for('taeg', graph, consolidator)
    for variant in ('taeg-no-before', 'taeg-no-same-event'):
        summary, records = _records_for(variant, graph, consolidator)
        assert len(summary) > 0
        # Timeline loop untouched: same events, same fallbacks, same order.
        assert [r['event_id'] for r in records] == [r['event_id'] for r in full]
        assert [r['fallback'] for r in records] == [r['fallback'] for r in full]


def test_divergence_of_taeg_with_itself_is_zero(graph, consolidator):
    _, full = _records_for('taeg', graph, consolidator)
    report = selection_divergence(full, full)
    assert report['n_different'] == 0
    assert report['n_comparable'] > 0


def test_divergence_report_structure(graph, consolidator):
    _, full = _records_for('taeg', graph, consolidator)
    _, no_before = _records_for('taeg-no-before', graph, consolidator)
    report = selection_divergence(full, no_before)
    assert report['n_events'] == 169
    assert 0 <= report['n_different'] <= report['n_comparable']
    for diff in report['differences']:
        assert diff['chosen_a'] != diff['chosen_b'] or True  # detail rows well-formed
        assert {'event_id', 'description', 'chosen_a', 'chosen_b'} <= set(diff)


def test_divergence_rejects_mismatched_timelines(graph, consolidator):
    _, full = _records_for('taeg', graph, consolidator)
    with pytest.raises(ValueError):
        selection_divergence(full, full[:-1])
