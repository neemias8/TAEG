"""
Tests for the timeline degradation experiment (JBCS revision, Task 4).
"""

import sys
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from degradation import consolidate_degraded, degrade_timeline  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def repo_cwd():
    old = os.getcwd()
    os.chdir(REPO_ROOT)
    yield
    os.chdir(old)


@pytest.fixture(scope="session")
def events():
    from data_loader import ChronologyLoader
    return ChronologyLoader().load_chronology()


def test_degrade_timeline_is_deterministic_and_ordered(events):
    kept1, removed1 = degrade_timeline(events, 0.25, seed=3)
    kept2, removed2 = degrade_timeline(events, 0.25, seed=3)
    kept3, _ = degrade_timeline(events, 0.25, seed=4)
    assert [e['id'] for e in kept1] == [e['id'] for e in kept2]
    assert removed1 == removed2
    assert [e['id'] for e in kept1] != [e['id'] for e in kept3]
    # Survivors keep chronological order.
    ids = [e['id'] for e in kept1]
    assert ids == sorted(ids)


def test_degrade_timeline_removal_counts(events):
    for fraction in (0.10, 0.25, 0.50):
        kept, removed = degrade_timeline(events, fraction, seed=0)
        assert len(removed) == round(len(events) * fraction)
        assert len(kept) + len(removed) == len(events)


def test_degrade_timeline_rejects_bad_fraction(events):
    with pytest.raises(ValueError):
        degrade_timeline(events, 1.0, seed=0)


def test_consolidate_degraded_excludes_removed_events():
    result = consolidate_degraded(0.25, seed=1)
    assert result['n_events_kept'] == 169 - round(169 * 0.25)
    record_ids = {r['event_id'] for r in result['records']}
    assert record_ids.isdisjoint(set(result['removed_event_ids']))
    assert len(result['records']) == result['n_events_kept']
    assert len(result['summary']) > 0
    assert result['centrality_info']['converged']


def test_consolidate_degraded_is_seed_reproducible():
    r1 = consolidate_degraded(0.10, seed=5)
    r2 = consolidate_degraded(0.10, seed=5)
    assert r1['summary'] == r2['summary']
    assert r1['removed_event_ids'] == r2['removed_event_ids']
