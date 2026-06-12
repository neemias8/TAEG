"""
Tests for the runner's reporting invariants (JBCS revision follow-up):
ordering-by-design verification and Kendall's Tau reporting convention.
"""

import sys
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session", autouse=True)
def repo_cwd():
    old = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def runner_module():
    import run_experiments
    return run_experiments


def test_event_order_monotonic_synthetic(runner_module):
    f = runner_module.event_order_monotonic
    assert f([{'event_id': 1}, {'event_id': 2}, {'event_id': 5}]) is True
    assert f([{'event_id': 2}, {'event_id': 1}]) is False
    assert f([{'event_id': 1}, {'event_id': 1}]) is False  # strictly increasing
    assert f([{'event_id': 7}]) is True
    assert f([]) is True


def test_apply_tau_by_design(runner_module):
    flat = {'rouge1_f1': 0.9, 'kendall_tau': 0.467}

    reported = runner_module.apply_tau_by_design(flat, monotonic=True)
    assert reported['kendall_tau'] == 1.0
    assert reported['tau_heuristic_matcher'] == 0.467
    assert reported['rouge1_f1'] == 0.9

    # A failed monotonicity check forbids the by-design claim: the heuristic
    # value stays in kendall_tau.
    unreported = runner_module.apply_tau_by_design(flat, monotonic=False)
    assert unreported['kendall_tau'] == 0.467
    assert unreported['tau_heuristic_matcher'] == 0.467

    # The input dict is never mutated.
    assert flat['kendall_tau'] == 0.467
    assert 'tau_heuristic_matcher' not in flat


def test_event_order_monotonic_on_real_runs(runner_module):
    """Every timeline-aware consolidation — full and degraded — must emit a
    strictly increasing event-ID sequence."""
    from improved_graph_builder import ImprovedTemporalGraphBuilder
    from selection_strategies import LongestStrategy, RandomStrategy
    from summarizer import LexRankTemporalAnchoring
    from degradation import consolidate_degraded

    graph = ImprovedTemporalGraphBuilder().build_improved_temporal_graph(verbose=False)
    ta = LexRankTemporalAnchoring()
    for strategy in (LongestStrategy(), RandomStrategy(seed=0)):
        _, records = ta.consolidate_with_strategy(strategy, graph=graph, verbose=False)
        assert runner_module.event_order_monotonic(records) is True

    degraded = consolidate_degraded(0.25, seed=0)
    assert runner_module.event_order_monotonic(degraded['records']) is True
