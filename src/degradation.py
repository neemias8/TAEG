"""
Timeline degradation experiment (JBCS revision, Task 4).

Randomly removes a fraction of events from the canonical timeline BEFORE
building the TAEG, runs the taeg strategy over the degraded graph, and the
result is evaluated against the FULL Golden Sample by the experiment runner.

Interpretation caveat (must accompany every report of these numbers):
removed events are simply absent from the output, so the experiment measures
completeness/content degradation; Kendall's Tau stays 1.0 among the
remaining events BY CONSTRUCTION of the timeline loop.
"""

import random
from typing import Dict, List, Tuple

from data_loader import ChronologyLoader
from selection_strategies import get_strategy
from summarizer import LexRankTemporalAnchoring
from taeg_centrality import build_graph_and_centrality

DEGRADATION_LEVELS = (0.10, 0.25, 0.50)
DEGRADATION_SEEDS = tuple(range(10))  # N=10 fixed, logged seeds per level


def degrade_timeline(events: List[Dict], fraction: float, seed: int) -> Tuple[List[Dict], List[int]]:
    """
    Remove ``fraction`` of the events uniformly at random (deterministic
    given ``seed``), preserving chronological order of the survivors.

    Returns (kept_events, removed_event_ids).
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1): {fraction}")
    rng = random.Random(seed)
    n_remove = round(len(events) * fraction)
    removed_ids = set(rng.sample([e['id'] for e in events], n_remove))
    kept = [e for e in events if e['id'] not in removed_ids]
    return kept, sorted(removed_ids)


def consolidate_degraded(fraction: float, seed: int,
                         strategy_key: str = 'taeg') -> Dict:
    """
    Run the taeg strategy over a degraded timeline.

    The TAEG is built only over the surviving events (BEFORE edges connect
    survivors that became consecutive), centrality is recomputed on the
    degraded graph, and the consolidation loop iterates the degraded
    timeline — removed events are absent from the output.
    """
    events = ChronologyLoader().load_chronology()
    kept, removed_ids = degrade_timeline(events, fraction, seed)
    graph, centrality, info = build_graph_and_centrality(events=kept, verbose=False)
    strategy = get_strategy(strategy_key, centrality=centrality)
    summary, records = LexRankTemporalAnchoring().consolidate_with_strategy(
        strategy, graph=graph, events=kept, verbose=False)
    return {
        'fraction': fraction,
        'seed': seed,
        'strategy': strategy_key,
        'n_events_total': len(events),
        'n_events_kept': len(kept),
        'removed_event_ids': removed_ids,
        'centrality_info': info,
        'summary': summary,
        'records': records,
    }
