from typing import Any, Dict, List, Set, Tuple

import networkx as nx

from ard.storage.graph import GraphBackend


class NetworkXBackend(GraphBackend):
    """
    NetworkX backend implementation for the knowledge graph.
    """

    def __init__(self):
        """Initialize a new NetworkX backend."""
        self._graph = nx.DiGraph()

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph):
        """Initialize a new NetworkX backend from a NetworkX graph."""
        backend = cls()
        backend._graph = graph
        return backend

    @classmethod
    def from_serializable(cls, data: Dict[str, Any]):
        """
        Initialize a new NetworkX backend from a serialized dictionary.

        Args:
            data (Dict[str, Any]): Serialized graph data from to_serializable()

        Returns:
            NetworkXBackend: New backend instance
        """
        backend = cls()

        # Add all nodes with their attributes
        for node_data in data["nodes"]:
            node_id = node_data["id"]
            attrs = node_data.get("attributes", {})
            backend.add_node(node_id, **attrs)

        # Add all edges with their attributes
        for edge_data in data["edges"]:
            source = edge_data["source"]
            target = edge_data["target"]
            attrs = edge_data.get("attributes", {})
            backend.add_edge(source, target, **attrs)

        return backend

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert the NetworkX graph to a serializable dictionary.

        Returns:
            Dict[str, Any]: A serializable representation of the graph
        """
        # Create serializable structures for nodes and edges
        nodes = []
        for node, attrs in self._graph.nodes(data=True):
            nodes.append({"id": node, "attributes": attrs})

        edges = []
        for source, target, attrs in self._graph.edges(data=True):
            edges.append({"source": source, "target": target, "attributes": attrs})

        return {"nodes": nodes, "edges": edges}

    def add_node(self, node: str, **attrs) -> None:
        """Add a node with optional attributes."""
        self._graph.add_node(node, **attrs)

    def has_node(self, node: str) -> bool:
        """Check if a node exists."""
        return self._graph.has_node(node)

    def add_edge(self, source: str, target: str, **attrs) -> None:
        """Add an edge with optional attributes."""
        self._graph.add_edge(source, target, **attrs)

    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists."""
        return self._graph.has_edge(source, target)

    def get_node_attrs(self, node: str) -> Dict[str, Any]:
        """Get all attributes of a node."""
        return dict(self._graph.nodes[node])

    def get_edge_attrs(self, source: str, target: str) -> Dict[str, Any]:
        """Get all attributes of an edge."""
        return dict(self._graph.edges[source, target])

    def get_nodes(self) -> Set[str]:
        """Get all nodes in the graph."""
        return set(self._graph.nodes())

    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all edges in the graph with their attributes."""
        return list(self._graph.edges(data=True))

    def get_successors(self, node: str) -> List[str]:
        """Get all successor nodes of a node."""
        return list(self._graph.successors(node))

    def get_predecessors(self, node: str) -> List[str]:
        """Get all predecessor nodes of a node."""
        return list(self._graph.predecessors(node))

    def get_out_edges(self, node: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all outgoing edges of a node with their attributes."""
        return list(self._graph.out_edges(node, data=True))

    def get_in_edges(self, node: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all incoming edges of a node with their attributes."""
        return list(self._graph.in_edges(node, data=True))

    def remove_node(self, node: str) -> None:
        """Remove a node and all its edges."""
        self._graph.remove_node(node)

    def number_of_edges(self) -> int:
        """Get the total number of edges in the graph."""
        return self._graph.number_of_edges()

    def shortest_path(
        self, source: str, target: str, directed: bool = True
    ) -> List[str]:
        """Get the shortest path between two nodes."""
        if directed:
            return nx.shortest_path(self._graph, source, target)
        else:
            return nx.shortest_path(self._graph.to_undirected(), source, target)

    def __len__(self) -> int:
        """Get the total number of nodes in the graph."""
        return len(self._graph)

    @property
    def graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph
