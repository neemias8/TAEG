"""
Graph builder module for TAEG project.

This module implements the core algorithm for constructing the Temporal Alignment 
Event Graph (TAEG) from the parsed XML data. The graph represents events as nodes
and temporal sequences as directed edges.

Author: Your Name
Date: September 2025
"""

import networkx as nx
import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import numpy as np

from .data_loader import DataLoader, Event

# Configure logging
logger = logging.getLogger(__name__)


@dataclass 
class GraphStatistics:
    """Statistics about the constructed graph."""
    num_nodes: int
    num_edges: int
    num_connected_components: int
    avg_degree: float
    max_degree: int
    gospel_coverage: Dict[str, int]  # Number of events per gospel
    edge_distribution: Dict[str, int]  # Number of edges per gospel


class TAEGGraphBuilder:
    """
    TAEG Graph Builder - Constructs temporal alignment event graphs.
    
    This class implements the core algorithm:
    1. Create nodes for each event with concatenated text from all gospels
    2. Create directed edges representing temporal sequences within each gospel
    3. Convert to torch_geometric format for GNN processing
    """
    
    def __init__(self, data_loader: DataLoader):
        """
        Initialize graph builder.
        
        Args:
            data_loader: Loaded DataLoader instance with events and gospel texts
        """
        self.data_loader = data_loader
        self.events = data_loader.events
        self.gospel_texts = data_loader.gospel_texts
        self.gospels = ['matthew', 'mark', 'luke', 'john']
        
        # Graph storage
        self.networkx_graph: Optional[nx.DiGraph] = None
        self.torch_geometric_data: Optional[Data] = None
        self.node_to_event_mapping: Dict[int, str] = {}
        self.event_to_node_mapping: Dict[str, int] = {}
        
    def build_graph(self, include_temporal_edges: bool = True) -> Data:
        """
        Build the complete TAEG graph.
        
        Args:
            include_temporal_edges: Whether to include temporal sequence edges
                                  (False for ablation study)
        
        Returns:
            torch_geometric.data.Data object representing the graph
        """
        logger.info("Building TAEG graph...")
        
        # Step 1: Create NetworkX graph
        self._create_networkx_graph()
        
        # Step 2: Add nodes with text content
        self._add_nodes_with_content()
        
        # Step 3: Add temporal edges (if enabled)
        if include_temporal_edges:
            self._add_temporal_edges()
        else:
            logger.info("Skipping temporal edges (ablation mode)")
        
        # Step 4: Convert to torch_geometric format
        self._convert_to_torch_geometric()
        
        # Step 5: Add node features (text embeddings will be added later)
        self._prepare_node_features()
        
        logger.info(f"Graph construction complete: {self.networkx_graph.number_of_nodes()} nodes, "
                   f"{self.networkx_graph.number_of_edges()} edges")
        
        return self.torch_geometric_data
    
    def _create_networkx_graph(self) -> None:
        """Initialize empty directed graph."""
        self.networkx_graph = nx.DiGraph()
        
        # Create mappings between events and node indices
        for i, event in enumerate(self.events):
            self.node_to_event_mapping[i] = event.event_id
            self.event_to_node_mapping[event.event_id] = i
    
    def _add_nodes_with_content(self) -> None:
        """Add nodes to graph with concatenated text content."""
        logger.info("Adding nodes with text content...")
        
        for i, event in enumerate(self.events):
            # Get concatenated text from all gospels for this event
            text_content = self.data_loader.get_concatenated_text_for_event(event)
            
            # Get list of gospels that mention this event
            participating_gospels = event.get_all_gospels_with_refs()
            
            # Add node with attributes
            self.networkx_graph.add_node(
                i,
                event_id=event.event_id,
                text_content=text_content,
                participating_gospels=participating_gospels,
                num_gospels=len(participating_gospels)
            )
    
    def _add_temporal_edges(self) -> None:
        """Add directed edges representing temporal sequences within each gospel."""
        logger.info("Adding temporal edges...")
        
        edge_count = 0
        
        for gospel in self.gospels:
            # Get sequence of events for this gospel
            event_sequence = self.data_loader.get_event_sequence_by_gospel(gospel)
            
            if len(event_sequence) < 2:
                continue
            
            # Add edges between consecutive events in this gospel
            for i in range(len(event_sequence) - 1):
                current_event_id = event_sequence[i]
                next_event_id = event_sequence[i + 1]
                
                # Get node indices
                current_node = self.event_to_node_mapping[current_event_id]
                next_node = self.event_to_node_mapping[next_event_id]
                
                # Add edge with gospel attribution
                if self.networkx_graph.has_edge(current_node, next_node):
                    # Edge already exists, add this gospel to the list
                    existing_gospels = self.networkx_graph[current_node][next_node]['gospels']
                    existing_gospels.append(gospel)
                    self.networkx_graph[current_node][next_node]['weight'] += 1
                else:
                    # Create new edge
                    self.networkx_graph.add_edge(
                        current_node,
                        next_node,
                        gospels=[gospel],
                        weight=1
                    )
                    edge_count += 1
        
        logger.info(f"Added {edge_count} unique temporal edges")
    
    def _convert_to_torch_geometric(self) -> None:
        """Convert NetworkX graph to torch_geometric format."""
        logger.info("Converting to torch_geometric format...")
        
        if self.networkx_graph.number_of_nodes() == 0:
            raise ValueError("Cannot convert empty graph")
        
        # Convert using torch_geometric utility
        self.torch_geometric_data = from_networkx(self.networkx_graph)
        
        # Add additional metadata
        self.torch_geometric_data.num_nodes = self.networkx_graph.number_of_nodes()
        self.torch_geometric_data.num_edges = self.networkx_graph.number_of_edges()
    
    def _prepare_node_features(self) -> None:
        """Prepare placeholder node features (actual embeddings added during training)."""
        num_nodes = self.torch_geometric_data.num_nodes
        
        # Create placeholder features (will be replaced with text embeddings)
        # For now, use simple numerical features
        node_features = []
        
        for i in range(num_nodes):
            event = self.events[i]
            
            # Feature vector: [num_gospels, matthew_present, mark_present, luke_present, john_present]
            features = [
                len(event.get_all_gospels_with_refs()),  # Total number of gospels
                1 if event.matthew_refs else 0,         # Matthew presence
                1 if event.mark_refs else 0,            # Mark presence  
                1 if event.luke_refs else 0,            # Luke presence
                1 if event.john_refs else 0             # John presence
            ]
            
            node_features.append(features)
        
        # Convert to tensor
        self.torch_geometric_data.x = torch.tensor(node_features, dtype=torch.float)
        
        # Store text content separately for embedding generation
        text_contents = []
        for i in range(num_nodes):
            text_contents.append(self.networkx_graph.nodes[i]['text_content'])
        
        self.torch_geometric_data.text_contents = text_contents
    
    def get_graph_statistics(self) -> GraphStatistics:
        """
        Compute and return graph statistics.
        
        Returns:
            GraphStatistics object with various metrics
        """
        if self.networkx_graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")
        
        G = self.networkx_graph
        
        # Basic statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        num_components = nx.number_weakly_connected_components(G)
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        # Gospel coverage (events per gospel)
        gospel_coverage = {}
        for gospel in self.gospels:
            count = 0
            for event in self.events:
                refs = getattr(event, f"{gospel}_refs")
                if refs:
                    count += 1
            gospel_coverage[gospel] = count
        
        # Edge distribution per gospel
        edge_distribution = {gospel: 0 for gospel in self.gospels}
        for u, v, data in G.edges(data=True):
            if 'gospels' in data:
                for gospel in data['gospels']:
                    edge_distribution[gospel] += 1
        
        return GraphStatistics(
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_connected_components=num_components,
            avg_degree=avg_degree,
            max_degree=max_degree,
            gospel_coverage=gospel_coverage,
            edge_distribution=edge_distribution
        )
    
    def visualize_graph_structure(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of graph structure.
        
        Args:
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.error("matplotlib not available for visualization")
            return
        
        if self.networkx_graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")
        
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.networkx_graph, k=1, iterations=50)
        
        # Color nodes by number of participating gospels
        node_colors = []
        for i in range(self.networkx_graph.number_of_nodes()):
            num_gospels = self.networkx_graph.nodes[i]['num_gospels']
            if num_gospels == 1:
                node_colors.append('lightblue')
            elif num_gospels == 2:
                node_colors.append('lightgreen')
            elif num_gospels == 3:
                node_colors.append('orange')
            else:  # 4 gospels
                node_colors.append('red')
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.networkx_graph, pos,
            node_color=node_colors,
            node_size=50,
            alpha=0.7
        )
        
        # Draw edges with different colors per gospel
        gospel_colors = {'matthew': 'blue', 'mark': 'green', 'luke': 'orange', 'john': 'purple'}
        
        for gospel in self.gospels:
            gospel_edges = []
            for u, v, data in self.networkx_graph.edges(data=True):
                if 'gospels' in data and gospel in data['gospels']:
                    gospel_edges.append((u, v))
            
            if gospel_edges:
                nx.draw_networkx_edges(
                    self.networkx_graph, pos,
                    edgelist=gospel_edges,
                    edge_color=gospel_colors[gospel],
                    alpha=0.5,
                    width=0.5
                )
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='1 Gospel'),
            mpatches.Patch(color='lightgreen', label='2 Gospels'),
            mpatches.Patch(color='orange', label='3 Gospels'),
            mpatches.Patch(color='red', label='4 Gospels')
        ]
        
        gospel_legend = [
            mpatches.Patch(color=gospel_colors[gospel], label=gospel.title())
            for gospel in self.gospels
        ]
        
        plt.legend(handles=legend_elements + gospel_legend, 
                  loc='upper right', bbox_to_anchor=(1.15, 1))
        
        plt.title("TAEG Graph Structure\nNodes colored by gospel participation")
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def _serialize_attribute(value: Any) -> Any:
        """Convert graph attributes into GraphML-friendly scalar types."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple, set)):
            converted = [TAEGGraphBuilder._serialize_attribute(v) for v in value]
            return ', '.join(str(v) for v in converted)
        if isinstance(value, dict):
            return json.dumps({str(k): TAEGGraphBuilder._serialize_attribute(v) for k, v in value.items()})
        return value

    def export_graph(self, filepath: str, format: str = 'graphml') -> None:
        """
        Export graph to file.
        
        Args:
            filepath: Output file path
            format: Export format ('graphml', 'gexf', 'edgelist')
        """
        if self.networkx_graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")
        
        sanitized_graph = self.networkx_graph.copy()

        for _, data in sanitized_graph.nodes(data=True):
            for key, value in list(data.items()):
                data[key] = self._serialize_attribute(value)

        for _, _, data in sanitized_graph.edges(data=True):
            for key, value in list(data.items()):
                data[key] = self._serialize_attribute(value)

        if format == 'graphml':
            nx.write_graphml(sanitized_graph, filepath)
        elif format == 'gexf':
            nx.write_gexf(sanitized_graph, filepath)
        elif format == 'edgelist':
            nx.write_edgelist(sanitized_graph, filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Graph exported to {filepath}")
    
    def get_node_neighborhoods(self, max_hops: int = 2) -> Dict[int, List[int]]:
        """
        Get k-hop neighborhoods for each node.
        
        Args:
            max_hops: Maximum number of hops to consider
            
        Returns:
            Dictionary mapping node_id to list of neighbor node_ids
        """
        if self.networkx_graph is None:
            raise ValueError("Graph not built yet. Call build_graph() first.")
        
        neighborhoods = {}
        
        for node in self.networkx_graph.nodes():
            # Get all nodes within max_hops distance
            neighbors = set()
            for hop in range(1, max_hops + 1):
                hop_neighbors = set()
                for target in self.networkx_graph.nodes():
                    try:
                        path_length = nx.shortest_path_length(
                            self.networkx_graph, node, target
                        )
                        if path_length == hop:
                            hop_neighbors.add(target)
                    except nx.NetworkXNoPath:
                        continue
                neighbors.update(hop_neighbors)
            
            neighborhoods[node] = list(neighbors)
        
        return neighborhoods


def main():
    """Example usage of TAEGGraphBuilder."""
    # Load data
    data_loader = DataLoader("data")
    events, gospel_texts = data_loader.load_all_data()
    
    if not events:
        logger.error("No events loaded. Please check data files.")
        return
    
    # Build graph
    graph_builder = TAEGGraphBuilder(data_loader)
    
    # Build complete graph
    torch_data = graph_builder.build_graph(include_temporal_edges=True)
    print(f"Complete graph: {torch_data.num_nodes} nodes, {torch_data.num_edges} edges")
    
    # Build ablation graph (no temporal edges)
    torch_data_ablation = graph_builder.build_graph(include_temporal_edges=False)
    print(f"Ablation graph: {torch_data_ablation.num_nodes} nodes, {torch_data_ablation.num_edges} edges")
    
    # Get statistics
    stats = graph_builder.get_graph_statistics()
    print("\nGraph Statistics:")
    print(f"  Nodes: {stats.num_nodes}")
    print(f"  Edges: {stats.num_edges}")
    print(f"  Connected components: {stats.num_connected_components}")
    print(f"  Average degree: {stats.avg_degree:.2f}")
    print(f"  Max degree: {stats.max_degree}")
    print(f"  Gospel coverage: {stats.gospel_coverage}")
    print(f"  Edge distribution: {stats.edge_distribution}")
    
    # Export graph
    graph_builder.export_graph("outputs/taeg_graph.graphml")


if __name__ == "__main__":
    main()
