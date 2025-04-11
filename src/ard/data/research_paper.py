import json
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

from ard.data.dataset_item import DataCategory, DatasetItem
from ard.data.metadata import Metadata, MetadataType


class ResearchPaper(DatasetItem):
    """
    Represents a scientific research paper in the dataset.
    Provides methods for accessing and managing paper-specific files.
    """

    def __init__(self, metadata: Metadata, storage_backend: Optional[str] = None):
        # Ensure the metadata type is set to PAPER
        if metadata.type != MetadataType.PAPER:
            metadata.type = MetadataType.PAPER

        super().__init__(metadata, storage_backend)

    def save_pdf(
        self, pdf_data: Union[bytes, BinaryIO], filename: Optional[str] = None
    ) -> str:
        """
        Save the PDF file for this research paper.

        Args:
            pdf_data: PDF content as bytes or file-like object
            filename: Optional filename to use (defaults to paper_id.pdf)

        Returns:
            str: The full path where the PDF was saved
        """
        if filename is None:
            filename = f"{self.id}.pdf"

        return self.save_file(filename, pdf_data, category=DataCategory.RAW.value)

    def get_pdf(self, filename: Optional[str] = None) -> bytes:
        """
        Retrieve the PDF file for this research paper.

        Args:
            filename: Optional filename to retrieve (defaults to paper_id.pdf)

        Returns:
            bytes: The PDF content
        """
        if filename is None:
            filename = f"{self.id}.pdf"

        return self.get_file(filename, category=DataCategory.RAW.value)

    def save_extracted_text(self, text: str, section: Optional[str] = None) -> str:
        """
        Save extracted text from the paper.

        Args:
            text: The extracted text content
            section: Optional section name (e.g., 'abstract', 'introduction')

        Returns:
            str: The full path where the text was saved
        """
        if section:
            filename = f"text/{section}.txt"
        else:
            filename = "text/full.txt"

        return self.save_file(
            filename, text.encode("utf-8"), category=DataCategory.PROCESSED.value
        )

    def get_extracted_text(self, section: Optional[str] = None) -> str:
        """
        Retrieve extracted text from the paper.

        Args:
            section: Optional section name (e.g., 'abstract', 'introduction')

        Returns:
            str: The extracted text content
        """
        if section:
            filename = f"text/{section}.txt"
        else:
            filename = "text/full.txt"

        return self.get_file(filename, category=DataCategory.PROCESSED.value).decode(
            "utf-8"
        )

    def list_extracted_sections(self) -> List[str]:
        """
        List all extracted text sections available for this paper.

        Returns:
            List[str]: List of section names
        """
        files = self.list_files(category=DataCategory.PROCESSED.value)
        sections = []

        for file_path in files:
            # Normalize path separators to forward slashes
            normalized_path = file_path.replace("\\", "/")
            if normalized_path.startswith("text/") and normalized_path.endswith(".txt"):
                section = Path(normalized_path).stem
                if section != "full":
                    sections.append(section)

        return sections

    def save_processed_data(self, data: Dict, name: str) -> str:
        """
        Save processed data as JSON.

        Args:
            data: Dictionary to save as JSON
            name: Name of the processed data file (without extension)

        Returns:
            str: The full path where the data was saved
        """
        filename = f"{name}.json"
        json_data = json.dumps(data, indent=2).encode("utf-8")
        return self.save_file(
            filename, json_data, category=DataCategory.PROCESSED.value
        )

    def get_processed_data(self, name: str) -> Dict:
        """
        Retrieve processed data from JSON.

        Args:
            name: Name of the processed data file (without extension)

        Returns:
            Dict: The loaded JSON data
        """
        filename = f"{name}.json"
        json_data = self.get_file(filename, category=DataCategory.PROCESSED.value)
        return json.loads(json_data.decode("utf-8"))

    def list_processed_data(self) -> List[str]:
        """
        List all processed data files available for this paper.

        Returns:
            List[str]: List of processed data file names (without extension)
        """
        files = self.list_files(category=DataCategory.PROCESSED.value)
        data_files = []

        for file_path in files:
            if file_path.endswith(".json") and not file_path.startswith("text/"):
                data_files.append(Path(file_path).stem)

        return data_files
