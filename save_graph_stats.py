#!/usr/bin/env python3
"""
Save graph statistics to a file for documentation purposes.
"""

import sys
from pathlib import Path
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph_builder_debug import TemporalGraphBuilder

def save_graph_stats():
    """Save graph statistics to a JSON file."""
    print("🔍 Building temporal graph and saving statistics...")

    builder = TemporalGraphBuilder()
    graph = builder.build_temporal_graph()

    # Save statistics to file
    stats_file = Path("outputs/graph_statistics.json")

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(graph['statistics'], f, indent=2, ensure_ascii=False)

    print(f"✅ Graph statistics saved to {stats_file}")

    # Also save a summary of edges for analysis
    edges_summary = {
        'before_edges': [edge for edge in graph['edges'] if edge['type'] == 'BEFORE'][:10],  # First 10
        'same_event_edges': [edge for edge in graph['edges'] if edge['type'] == 'SAME_EVENT'][:20],  # First 20
        'total_before': len([e for e in graph['edges'] if e['type'] == 'BEFORE']),
        'total_same_event': len([e for e in graph['edges'] if e['type'] == 'SAME_EVENT'])
    }

    edges_file = Path("outputs/graph_edges_summary.json")
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump(edges_summary, f, indent=2, ensure_ascii=False)

    print(f"✅ Graph edges summary saved to {edges_file}")

if __name__ == "__main__":
    save_graph_stats()