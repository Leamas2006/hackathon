import re

SUBGRAPH_FILE_EXTENSION = ".subgraph.json"


def normalize_file_name(file_name: str) -> str:
    """Normalize file name to ensure it's a valid S3 key."""
    # Replace special characters with underscores
    normalized = re.sub(r"[\\/ ]", "_", file_name)
    return normalized


def get_subgraph_file_name(subgraph_name: str) -> str:
    return f"{subgraph_name}{SUBGRAPH_FILE_EXTENSION}"


def get_subgraph_name(subgraph_path: str) -> str:
    subgraph_filename = subgraph_path.split("/")[-1]
    return subgraph_filename.split(SUBGRAPH_FILE_EXTENSION)[0]
