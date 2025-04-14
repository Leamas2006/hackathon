import os
import json
import random
import networkx as nx

from ard.api.openai_client import OpenAIClient
from ard.api.firecrawl_client import FirecrawlClient

from ard.hypothesis.hypothesis import Hypothesis
from ard.hypothesis.saver import HypothesisSaver, JSONParser
from ard.storage.file import StorageManager

# 🔐 Your API keys
OPENAI_API_KEY = "sk-proj-aXvv31x1jAZb-onjHmd7lxy_eHGeYiAFcecJ8YD8qr_iOlw-QN2-o0qdNvMzGJhsmOcy2KBx|WT3B|bkFJkW6NRhd1ENо_olZvSXHAfJgi8J_vY1о2vTler0zFZ5XWBv6QUWAHо1Lbu0iDhq5wMqqPgUSggА"
FIRECRAWL_API_KEY = "yfc-d8b5246b3b744b1aa3d0a269948a5d08"

# 📂 Path to subgraph JSON
current_dir = os.path.dirname(__file__)
graph_path = os.path.abspath(os.path.join(current_dir, "../data/sample_subgraph.json"))

# 🔧 Load and build graph
def load_graph(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def build_graph(data):
    G = nx.Graph()
    for node in data.get("nodes", []):
        node_id = node.get("id", node.get("name"))
        if node_id:
            G.add_node(node_id, **{k: v for k, v in node.items() if k != "id"})
    for edge in data.get("edges", []):
        src = edge.get("source", edge.get("from"))
        tgt = edge.get("target", edge.get("to"))
        if src and tgt:
            G.add_edge(src, tgt, **{k: v for k, v in edge.items() if k not in ["source", "target", "from", "to"]})
    return G

def prepare_prompt(subgraph):
    node_lines = "\n".join(f"- {n}" for n in subgraph.nodes)
    edge_lines = "\n".join(
        f"- {u} - {v} ({subgraph[u][v].get('relation', 'related to')})"
        for u, v in subgraph.edges
    )
    return (
        f"The following subgraph contains:\n\n"
        f"Nodes:\n{node_lines}\n\n"
        f"Edges:\n{edge_lines}\n\n"
        f"Based on the structure above, generate a plausible scientific hypothesis."
    )

# 🔁 Main execution
def main():
    data = load_graph(graph_path)
    graph = build_graph(data)

    if len(graph.nodes) < 4:
        print("[Error] Graph must contain at least 4 nodes.")
        return

    sub_nodes = random.sample(list(graph.nodes), 4)
    subgraph = graph.subgraph(sub_nodes)

    prompt = prepare_prompt(subgraph)

    openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
    firecrawl_client = FirecrawlClient(api_key=FIRECRAWL_API_KEY)

    try:
        hypothesis_text = openai_client.generate_hypothesis_from_prompt(prompt)
    except Exception as e:
        print(f"[OpenAI Error] {e}")
        return

    try:
        search_results = firecrawl_client.search(hypothesis_text, limit=3)
        references = [item.get("url") or item.get("title") for item in search_results.get("results", [])]
    except Exception as e:
        print(f"[Firecrawl Error] {e}")
        references = []

    # 🧠 Create Hypothesis object
    hypothesis = Hypothesis(
        title="Generated Hypothesis",
        statement=hypothesis_text,
        source=subgraph,  # requires Subgraph class, assumed imported
        method=openai_client,
        references=references
    )

    # 💾 Save it
    saver = HypothesisSaver(
        storage_backend=StorageManager(base_path="results/"),
        parser=JSONParser()
    )
    hypothesis.save(saver=saver)

    print("\n✅ Hypothesis saved successfully!")

if __name__ == "__main__":
    main()








