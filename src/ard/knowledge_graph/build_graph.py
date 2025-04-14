import networkx as nx

def build_graph_from_data(data):
    """Builds graph from dict structure like in your example."""
    graph_data = data.get("graph_data", {})
    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", [])

    G = nx.DiGraph()

    for node_id, node_info in nodes.items():
        G.add_node(node_id, **node_info)

    for edge in edges:
        G.add_edge(
            edge["source"],
            edge["target"],
            relation=edge.get("relation", ""),
            sources=edge.get("sources", [])
        )

    return G

