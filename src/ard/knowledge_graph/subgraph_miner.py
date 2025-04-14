# src/ard/knowledge_graph/subgraph_miner.py

import networkx as nx
import random

def mine_random_subgraph(G, size=5):
    """Return a random subgraph of specified size."""
    nodes = random.sample(list(G.nodes), size)
    return G.subgraph(nodes).copy()
