import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ard.storage.file import StorageBackend
from ard.storage.file.utils import normalize_file_name

if TYPE_CHECKING:
    from ard.hypothesis.hypothesis import Hypothesis


class Parser(Protocol):
    def parse(self, hypothesis: "Hypothesis"): ...

    output_type: str


@dataclass
class HypothesisSaver:
    """A class for saving hypotheses to a storage backend."""

    storage_backend: "StorageBackend"
    parser: Parser

    def save(self, hypothesis: "Hypothesis") -> None:
        """Save a hypothesis to the storage backend."""
        parsed_hypothesis = self.parser.parse(hypothesis)

        file_name = self.get_file_name(hypothesis)

        self.storage_backend.save_file(
            file_name,
            f"{file_name}.{self.parser.output_type}",
            bytes(parsed_hypothesis, "utf-8"),
        )

    def get_file_name(self, hypothesis: "Hypothesis") -> str:
        file_name = normalize_file_name(hypothesis.title)

        if len(file_name) > 100:
            file_name = f"{file_name[:100]}[...truncated]"

        return file_name


class MarkdownParser(Parser):
    output_type = "md"

    def parse(self, hypothesis: "Hypothesis") -> str:
        return f"""
# {hypothesis.title}

{hypothesis.statement}

## References
{"\n".join(hypothesis.references)}

## Context
{hypothesis.source._context}

## Subgraph
```cypher
{hypothesis.source.to_cypher_string()}
```
"""


class JSONParser(Parser):
    output_type = "json"

    def parse(self, hypothesis: "Hypothesis"):
        return json.dumps(
            {
                "title": hypothesis.title,
                "text": hypothesis.statement,
                "source": hypothesis.source.to_json(),
                "metadata": hypothesis.metadata,
                "method_name": str(hypothesis.method),
                "method": hypothesis.method.to_json(),
                "references": hypothesis.references,
            }
        )
