"""
Tests for the selection-level evaluation (JBCS revision, Task 5b).
"""

import sys
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from selection_eval import (  # noqa: E402
    SelectionEvaluator,
    parse_golden_segments,
    percentile_in_distribution,
)


@pytest.fixture(scope="session", autouse=True)
def repo_cwd():
    old = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def golden_text():
    return (REPO_ROOT / "data" / "Golden_Sample.txt").read_text(encoding="utf-8").strip()


@pytest.fixture(scope="session")
def graph():
    from improved_graph_builder import ImprovedTemporalGraphBuilder
    return ImprovedTemporalGraphBuilder().build_improved_temporal_graph(verbose=False)


@pytest.fixture(scope="session")
def sel_eval(graph, golden_text):
    return SelectionEvaluator(graph, golden_text)


def test_golden_parser_validates_known_quirks(golden_text):
    segments, report = parse_golden_segments(golden_text)
    # Known data quirks (documented in the spec): marker 169 is absent and
    # segment 162 is empty. Everything else must be present and non-empty.
    assert report['markers_found'] == 168
    assert report['missing_markers'] == [169]
    assert report['empty_segments'] == [162]
    assert segments[1].startswith("Six days before the Passover")
    # Segments must jointly cover almost the whole Golden Sample.
    assert report['covered_chars'] > 0.98 * report['golden_chars'] - 1000


def test_contested_set_matches_spec_counts(sel_eval):
    # 96 contested events by chronology references: 21 with 2 versions,
    # 51 with 3, 24 with 4 (docs/JBCS_REVISION_SPEC.md, Task 5b).
    assert len(sel_eval.contested_ids) == 96
    # Analytical random floor (21/2 + 51/3 + 24/4)/96 ~= 0.349
    assert abs(sel_eval.analytical_floor - (21 / 2 + 51 / 3 + 24 / 4) / 96) < 1e-9
    # Every contested event is either evaluated or excluded with a reason.
    assert len(sel_eval.oracles) + len(sel_eval.excluded) == 96
    assert len(sel_eval.oracles) > 0


def test_oracle_is_argmax_and_deterministic(sel_eval, graph, golden_text):
    again = SelectionEvaluator(graph, golden_text)
    assert {k: o['oracle_node'] for k, o in sel_eval.oracles.items()} == \
           {k: o['oracle_node'] for k, o in again.oracles.items()}
    for o in sel_eval.oracles.values():
        assert o['scores'][o['oracle_node']] == max(o['scores'].values())


def _records_choosing(sel_eval, pick):
    """Synthetic records: pick(oracle_info) -> node_id for evaluated events."""
    return [
        {'event_id': eid, 'fallback': False, 'chosen_node': pick(o)}
        for eid, o in sel_eval.oracles.items()
    ]


def test_oracle_accuracy_bounds(sel_eval):
    always = _records_choosing(sel_eval, lambda o: o['oracle_node'])
    acc = sel_eval.oracle_accuracy(always)
    assert acc['accuracy'] == 1.0
    assert acc['n_evaluated'] == len(sel_eval.oracles)

    never = _records_choosing(
        sel_eval,
        lambda o: next(n for n in o['scores'] if n != o['oracle_node']))
    assert sel_eval.oracle_accuracy(never)['accuracy'] < 1.0


def test_contested_subset_texts_alignment(sel_eval):
    records = _records_choosing(sel_eval, lambda o: o['oracle_node'])
    hyp, ref = sel_eval.contested_subset_texts(records)
    assert len(hyp) > 0 and len(ref) > 0
    # The reference side is exactly the golden segments of evaluated events.
    first = sel_eval.segments[sorted(sel_eval.oracles)[0]]
    assert ref.startswith(first[:50])


def test_percentile_in_distribution():
    dist = [0.1, 0.2, 0.3, 0.4]
    assert percentile_in_distribution(0.05, dist) == 0.0
    assert percentile_in_distribution(0.25, dist) == 50.0
    assert percentile_in_distribution(0.4, dist) == 100.0
    with pytest.raises(ValueError):
        percentile_in_distribution(0.5, [])
