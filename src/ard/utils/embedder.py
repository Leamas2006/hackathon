import json
import os
from typing import Dict, List

import numpy as np


class Embedder:
    """
    A class for generating, storing, and retrieving embeddings for nodes in a knowledge graph.

    This class provides a centralized way to handle embeddings, which can be used by
    various components like node mergers and path generators.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
        distance_metric: str = "cosine",
    ):
        """
        Initialize the Embedder.

        Args:
            model_name: Name of the SentenceTransformer model to use
            cache_embeddings: Whether to cache embeddings in memory
            distance_metric: Distance metric to use ('cosine', 'euclidean', or 'dot')
        """
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.distance_metric = distance_metric
        self._model = None  # Lazy-loaded
        self._embeddings = {}  # Cache for embeddings

    def _load_model(self):
        """
        Lazy-load the embedding model.

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "The sentence-transformers package is required for Embedder. "
                    "Install it with 'pip install sentence-transformers'"
                )

            self._model = SentenceTransformer(self.model_name)

    def embed(
        self,
        words: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Calculate embeddings for all nodes in the knowledge graph.

        Args:
            words: The words to embed

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping node names to their embeddings
        """
        self._load_model()

        # Get all nodes from the graph
        embeddings = self._model.encode(words, show_progress_bar=False)

        # Ensure embeddings are 2D array
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)

        # Store embeddings in cache if enabled
        if self.cache_embeddings:
            for word, embedding in zip(words, embeddings):
                self._embeddings[word] = embedding.flatten()  # Store as 1D array

        return {word: embedding.flatten() for word, embedding in zip(words, embeddings)}

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get the embedding for a given text.

        If the embedding is cached, return it. Otherwise, calculate it,
        cache it if caching is enabled, and return it.

        Args:
            text: The text to get the embedding for

        Returns:
            np.ndarray: The embedding vector
        """
        # Check if embedding is in cache
        if text in self._embeddings:
            return self._embeddings[text]

        # Calculate embedding
        self._load_model()
        embedding = self._model.encode(text, show_progress_bar=False)

        # Ensure embedding is 1D
        embedding = np.asarray(embedding).flatten()

        # Cache embedding if enabled
        if self.cache_embeddings:
            self._embeddings[text] = embedding

        return embedding

    def get_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of texts to get embeddings for

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping texts to their embeddings
        """
        # Check which texts are not in cache
        missing_texts = [text for text in texts if text not in self._embeddings]

        # Calculate embeddings for missing texts
        if missing_texts:
            self._load_model()
            missing_embeddings = self._model.encode(
                missing_texts, show_progress_bar=False
            )

            # Ensure embeddings are 2D array
            if len(missing_embeddings.shape) == 1:
                missing_embeddings = missing_embeddings.reshape(1, -1)

            # Cache embeddings if enabled
            if self.cache_embeddings:
                for text, embedding in zip(missing_texts, missing_embeddings):
                    self._embeddings[text] = embedding.flatten()

        # Return embeddings for all texts
        return {text: self.get_embedding(text) for text in texts}

    @property
    def embeddings_len(self) -> int:
        return len(self._embeddings)

    def calculate_distance(self, text1: str, text2: str, metric: str = None) -> float:
        """
        Calculate the distance between two texts.

        Args:
            text1: First text
            text2: Second text
            metric: Distance metric to use ('cosine', 'euclidean', or 'dot'). If None,
                   uses the metric specified during initialization.

        Returns:
            float: Distance between the two texts

        Raises:
            ValueError: If the specified metric is not supported
        """
        # Get embeddings
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)

        # Use provided metric or fall back to default
        metric = metric or self.distance_metric

        # Calculate distance based on selected metric
        if metric == "cosine":
            return 1.0 - self._cosine_similarity(embedding1, embedding2)
        elif metric == "euclidean":
            return self._euclidean_distance(embedding1, embedding2)
        elif metric == "dot":
            return -self._dot_product(
                embedding1, embedding2
            )  # Negative because smaller is better for distance
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def calculate_similarity(self, text1: str, text2: str, metric: str = None) -> float:
        """
        Calculate the similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            metric: Distance metric to use ('cosine', 'euclidean', or 'dot'). If None,
                   uses the metric specified during initialization.

        Returns:
            float: Similarity between the two texts (higher is more similar)

        Raises:
            ValueError: If the specified metric is not supported
        """
        # Get embeddings
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)

        # Use provided metric or fall back to default
        metric = metric or self.distance_metric

        # Calculate similarity based on selected metric
        if metric == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        elif metric == "euclidean":
            # Convert euclidean distance to similarity (1 / (1 + distance))
            distance = self._euclidean_distance(embedding1, embedding2)
            return 1.0 / (1.0 + distance)
        elif metric == "dot":
            return self._dot_product(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Cosine similarity (0-1)
        """
        # Ensure vectors are 1D
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()

        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Euclidean distance
        """
        # Ensure vectors are 1D
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()

        return np.linalg.norm(vec1 - vec2)

    def _dot_product(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate dot product between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Dot product
        """
        # Ensure vectors are 1D
        vec1 = np.asarray(vec1).flatten()
        vec2 = np.asarray(vec2).flatten()

        return np.dot(vec1, vec2)

    def save_to_file(self, filename: str) -> None:
        """
        Save embeddings to a JSON file.

        Args:
            filename: Path to save the embeddings to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_embeddings = {}
        for key, embedding in self._embeddings.items():
            serializable_embeddings[key] = embedding.tolist()

        # Save embeddings and metadata
        data = {
            "model_name": self.model_name,
            "distance_metric": self.distance_metric,
            "embeddings": serializable_embeddings,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename: str) -> None:
        """
        Load embeddings from a JSON file.

        Args:
            filename: Path to load the embeddings from

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file contains incompatible data
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Embeddings file not found: {filename}")

        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading embeddings file: {str(e)}")

        # Check if the model name matches
        if data["model_name"] != self.model_name:
            raise ValueError(
                f"Model name mismatch: {data['model_name']} (file) vs {self.model_name} (current)"
            )

        # Convert list embeddings back to numpy arrays
        self._embeddings = {}
        for key, embedding_list in data["embeddings"].items():
            self._embeddings[key] = np.array(embedding_list)

        # Update distance metric if it exists in the file
        if "distance_metric" in data:
            self.distance_metric = data["distance_metric"]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embeddings = {}
