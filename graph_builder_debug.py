#!/usr/bin/env python3
"""
Graph Builder Module for TAEG - Temporal Anchoring Graph Generation
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import xml.etree.ElementTree as ET

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import ChronologyLoader, BiblicalDataLoader


class TemporalGraphBuilder:
    """Builder for temporal graphs with debugging capabilities."""

    def __init__(self):
        """Initialize the graph builder."""
        self.chrono_loader = ChronologyLoader()
        self.biblical_loader = BiblicalDataLoader()

    def build_temporal_graph(self) -> Dict[str, Any]:
        """
        Build temporal graph from chronology XML with proper SAME_EVENT and BEFORE edges.

        Returns:
            Dictionary representing the temporal graph with statistics
        """
        print("🔍 Loading chronology data...")
        events = self.chrono_loader.load_chronology()
        print(f"✅ Loaded {len(events)} chronological events")

        graph = {
            'nodes': {},
            'edges': [],
            'statistics': {
                'total_events': len(events),
                'total_nodes': 0,
                'before_edges': 0,
                'same_event_edges': 0,
                'events_with_multiple_gospels': 0,
                'gospel_distribution': {'matthew': 0, 'mark': 0, 'luke': 0, 'john': 0}
            }
        }

        # Create nodes for each event
        print("🏗️ Creating event nodes...")
        for event in events:
            event_id = event['id']
            node_id = f"event_{event_id}"

            # Count gospel mentions
            gospel_mentions = []
            for gospel in ['matthew', 'mark', 'luke', 'john']:
                if event.get(gospel) and event[gospel].strip():
                    gospel_mentions.append(gospel)
                    graph['statistics']['gospel_distribution'][gospel] += 1

            # Extract texts for this event
            event_texts = self._extract_event_text(event)

            graph['nodes'][node_id] = {
                'event': event,
                'texts': event_texts,
                'gospels': gospel_mentions,
                'text_count': len(event_texts)
            }

            if len(gospel_mentions) > 1:
                graph['statistics']['events_with_multiple_gospels'] += 1

        graph['statistics']['total_nodes'] = len(graph['nodes'])
        print(f"✅ Created {len(graph['nodes'])} nodes")

        # Create BEFORE edges between consecutive events
        print("🔗 Creating BEFORE edges...")
        event_ids = sorted([node['event']['id'] for node in graph['nodes'].values()])

        for i in range(len(event_ids) - 1):
            current_id = event_ids[i]
            next_id = event_ids[i + 1]

            current_node = f"event_{current_id}"
            next_node = f"event_{next_id}"

            graph['edges'].append({
                'from': current_node,
                'to': next_node,
                'type': 'BEFORE',
                'description': f"Event {current_id} → Event {next_id}"
            })
            graph['statistics']['before_edges'] += 1

        print(f"✅ Created {graph['statistics']['before_edges']} BEFORE edges")

        # Create SAME_EVENT edges for events mentioned in multiple gospels
        print("🔗 Creating SAME_EVENT edges...")
        same_event_pairs = self._find_same_event_pairs(events)

        for pair in same_event_pairs:
            event1_id = f"event_{pair[0]}"
            event2_id = f"event_{pair[1]}"

            # Only add if both nodes exist
            if event1_id in graph['nodes'] and event2_id in graph['nodes']:
                graph['edges'].append({
                    'from': event1_id,
                    'to': event2_id,
                    'type': 'SAME_EVENT',
                    'description': f"Same event: {pair[0]} ↔ {pair[1]}"
                })
                graph['statistics']['same_event_edges'] += 1

        print(f"✅ Created {graph['statistics']['same_event_edges']} SAME_EVENT edges")

        # Print detailed statistics
        self._print_graph_statistics(graph)

        return graph

    def _find_same_event_pairs(self, events: List[Dict]) -> List[Tuple[int, int]]:
        """
        Find pairs of events that represent the same chronological event
        mentioned in different gospels.

        Strategy: Look for events that are very close chronologically and have
        complementary gospel coverage (different gospels mentioning them).
        """
        pairs = []

        # Group events by chronological proximity (within 3 events)
        for i, event1 in enumerate(events):
            for j in range(i + 1, min(i + 4, len(events))):  # Check next 3 events
                event2 = events[j]

                # Get gospels mentioning each event
                gospels1 = set(g for g in ['matthew', 'mark', 'luke', 'john']
                              if event1.get(g) and event1[g].strip())
                gospels2 = set(g for g in ['matthew', 'mark', 'luke', 'john']
                              if event2.get(g) and event2[g].strip())

                # If they have complementary gospel coverage (some overlap or different gospels)
                if gospels1 and gospels2 and (gospels1 & gospels2 or gospels1 != gospels2):
                    # Check if descriptions are reasonably similar or if they're very close chronologically
                    if (self._descriptions_similar(event1['description'], event2['description'], threshold=0.3) or
                        abs(event1['id'] - event2['id']) <= 2):

                        pairs.append((event1['id'], event2['id']))
                        print(f"🔗 Found SAME_EVENT pair: Event {event1['id']} '{event1['description']}' ({gospels1}) ↔ Event {event2['id']} '{event2['description']}' ({gospels2})")

        return pairs

    def _descriptions_similar(self, desc1: str, desc2: str, threshold: float = 0.6) -> bool:
        """Check if two descriptions are similar using simple text overlap."""
        if not desc1 or not desc2:
            return False

        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union)
        return similarity >= threshold

    def _extract_event_text(self, event: Dict) -> List[str]:
        """Extract text content for an event from biblical texts."""
        texts = []

        # For each gospel that mentions this event
        for gospel in ['matthew', 'mark', 'luke', 'john']:
            reference = event.get(gospel)
            if reference and reference.strip():
                try:
                    # Load the full gospel text and extract relevant verses
                    gospel_text = self.biblical_loader.load_gospel_text(gospel.capitalize())

                    # For now, return the full passion week text for this gospel
                    # In a more sophisticated implementation, you'd extract specific verses
                    if gospel_text:
                        texts.append(gospel_text)

                except Exception as e:
                    print(f"Warning: Could not extract text for {gospel} {reference}: {e}")

        return texts

    def _print_graph_statistics(self, graph: Dict[str, Any]):
        """Print detailed statistics about the graph."""
        stats = graph['statistics']

        print("\n" + "="*60)
        print("📊 GRAPH STATISTICS")
        print("="*60)

        print(f"📅 Total Events: {stats['total_events']}")
        print(f"🏗️ Total Nodes: {stats['total_nodes']}")
        print(f"🔗 BEFORE Edges: {stats['before_edges']}")
        print(f"🔄 SAME_EVENT Edges: {stats['same_event_edges']}")
        print(f"📚 Events with Multiple Gospels: {stats['events_with_multiple_gospels']}")

        print("\n📖 Gospel Distribution:")
        for gospel, count in stats['gospel_distribution'].items():
            print(f"  {gospel.capitalize()}: {count} mentions")

        print("\n🔍 Sample Nodes:")
        for i, (node_id, node_data) in enumerate(list(graph['nodes'].items())[:5]):
            event = node_data['event']
            print(f"  {node_id}: '{event['description']}' (Gospels: {', '.join(node_data['gospels'])})")

        print("\n🔗 Sample Edges:")
        for edge in graph['edges'][:10]:
            print(f"  {edge['from']} --[{edge['type']}]--> {edge['to']}")

        if len(graph['edges']) > 10:
            print(f"  ... and {len(graph['edges']) - 10} more edges")

        print("="*60)


def main():
    """Main function to build and analyze the temporal graph."""
    print("🚀 TAEG - Temporal Graph Builder")
    print("="*60)

    builder = TemporalGraphBuilder()
    graph = builder.build_temporal_graph()

    print("\n✅ Graph construction completed!")
    print(f"📊 Graph has {graph['statistics']['total_nodes']} nodes and {len(graph['edges'])} edges")


if __name__ == "__main__":
    main()