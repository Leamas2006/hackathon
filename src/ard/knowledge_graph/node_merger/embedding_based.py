from typing import List, Set

from ard.knowledge_graph.node_merger.base import NodeMerger
from ard.utils.embedder import Embedder


class EmbeddingBasedNodeMerger(NodeMerger):
    """
    Merges nodes based on embedding similarity.

    This merger uses a sentence transformer model to generate embeddings for node names
    and identifies nodes with similarity above a threshold for merging.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize the embedding-based node merger.

        Args:
            embedding_model_name: Name of the SentenceTransformer model to use
            similarity_threshold: Threshold for cosine similarity (0-1)
        """
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold = similarity_threshold
        self.embedder = Embedder(
            model_name=embedding_model_name, distance_metric="cosine"
        )

    def find_merge_candidates(self, knowledge_graph) -> List[Set[str]]:
        """
        Find groups of nodes with similar embeddings.

        Args:
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            List[Set[str]]: List of sets where each set contains node names to be merged
        """
        # Find groups of similar nodes
        similar_groups = []
        processed_nodes = set()

        for node1 in knowledge_graph.get_nodes():
            if node1 in processed_nodes:
                continue

            similar_nodes = {node1}
            for node2 in knowledge_graph.get_nodes():
                if node1 == node2 or node2 in processed_nodes:
                    continue

                similarity = self.embedder.calculate_similarity(node1, node2)
                if similarity >= self.similarity_threshold:
                    similar_nodes.add(node2)

            if len(similar_nodes) > 1:
                similar_groups.append(similar_nodes)
                processed_nodes.update(similar_nodes)

        return similar_groups

    def generate_merged_node_name(self, nodes: Set[str], knowledge_graph) -> str:
        """
        Generate a name for the merged node, using the most frequent name.

        Args:
            nodes: Set of node names to be merged
            knowledge_graph: The KnowledgeGraph instance

        Returns:
            str: The name for the merged node
        """
        # Use the most frequently used node name
        node_counts = {}
        for node in nodes:
            node_attrs = knowledge_graph.get_node_attrs(node)
            count = len(node_attrs.get("sources", []))
            node_counts[node] = count

        # Return the node with highest count
        return max(node_counts.items(), key=lambda x: x[1])[0]
