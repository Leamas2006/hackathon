import os
from typing import TYPE_CHECKING, Dict, List, Optional

from loguru import logger

from ard.data.dataset_item import DatasetItem
from ard.storage.file import LocalStorageBackend, StorageManager

if TYPE_CHECKING:
    from ard.data.triplets import Triplets


class Dataset:
    """
    A collection of DatasetItem objects representing a dataset.

    The Dataset class provides a container for managing and accessing a collection
    of DatasetItem objects. It supports initialization with a list of items,
    loading from a local directory, and standard container operations.

    Attributes:
        items (List[DatasetItem]): The list of DatasetItem objects in the dataset.
    """

    def __init__(self, items: List[DatasetItem]) -> None:
        """
        Initialize a Dataset with a list of DatasetItem objects.

        Args:
            items (List[DatasetItem]): A list of DatasetItem objects to include in the dataset.
        """
        self.items = items

    @classmethod
    def from_local(cls, path: str) -> "Dataset":
        """
        Create a Dataset by loading DatasetItem objects from a local directory.

        This method scans the specified directory and attempts to create a DatasetItem
        for each item found. Items that cannot be loaded (e.g., due to missing metadata)
        are skipped with a warning.

        Args:
            path (str): The path to the directory containing dataset items.

        Returns:
            Dataset: A new Dataset instance containing the successfully loaded items.
        """
        # register the local storage backend
        storage_manager = StorageManager()
        storage_manager.register_backend("local", LocalStorageBackend(path))

        items = []
        for item in os.listdir(path):
            # skip files
            if os.path.isfile(os.path.join(path, item)):
                continue
            try:
                items.append(DatasetItem.from_local(item))
            except FileNotFoundError:
                logger.warning(f"Item {item} not found, skipping")

        return cls(items)

    def get_triplets(
        self,
        kg_version: Optional[str] = None,
        build_graph: bool = False,
        skip_errors: bool = True,
    ) -> Dict[str, "Triplets"]:
        """
        Get triplets from all items in the dataset.

        This method retrieves triplets from each DatasetItem in the dataset.
        If an item doesn't have the specified KG version (or any KG version if none is specified),
        it will be skipped if skip_errors is True, otherwise an error will be raised.

        Args:
            kg_version: Optional name of the KG version (e.g., 'baseline_1').
                        If not provided, the latest version for each item will be used.
            build_graph: Whether to build the graph during initialization.
                        If False, the graph will be built on-demand when needed.
            skip_errors: Whether to skip items that don't have the specified KG version
                        or any KG version if none is specified.

        Returns:
            Dict[str, Triplets]: A dictionary mapping item IDs to their Triplets objects

        Raises:
            FileNotFoundError: If skip_errors is False and an item doesn't have the specified KG version
                              or any KG version if none is specified.
        """

        result = {}

        for item in self.items:
            try:
                triplets = item.get_triplets(
                    kg_version=kg_version, build_graph=build_graph
                )
                result[item.id] = triplets
            except (FileNotFoundError, ValueError) as e:
                if skip_errors:
                    logger.warning(
                        f"Could not get triplets for item {item.id}: {str(e)}"
                    )
                else:
                    raise

        return result

    def __repr__(self) -> str:
        """
        Return a string representation of the Dataset for debugging.

        Returns:
            str: A string representation of the Dataset, including its items.
        """
        return f"Dataset(items={self.items})"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Dataset.

        Returns:
            str: A string representation of the Dataset, including its items.
        """
        return f"Dataset(items={self.items})"

    def __len__(self) -> int:
        """
        Return the number of items in the Dataset.

        Returns:
            int: The number of DatasetItem objects in the dataset.
        """
        return len(self.items)
