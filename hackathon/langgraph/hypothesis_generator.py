import re
from typing import Any

from langchain_core.runnables import RunnableConfig
from langfuse.callback import CallbackHandler

from ard.hypothesis import Hypothesis, HypothesisGeneratorProtocol
from ard.subgraph import Subgraph

from .graph import hypgen_graph
from .state import HypgenState
from .utils import message_to_dict

langfuse_callback = CallbackHandler()


class HypothesisGenerator(HypothesisGeneratorProtocol):
    def run(self, subgraph: Subgraph) -> Hypothesis:
        context = subgraph.context
        path = subgraph.to_cypher_string(full_graph=False)

        res: HypgenState = hypgen_graph.invoke(
            {"subgraph": path, "context": context},
            config=RunnableConfig(callbacks=[langfuse_callback], recursion_limit=100),
        )

        title = self.__parse_title(res) or ""
        statement = self.__parse_statement(res)
        return Hypothesis(
            title=title,
            statement=statement,
            source=subgraph,
            method=self,
            metadata={
                "summary": res["summary"],
                "context": res["context"],
                "novelty": res["novelty"],
                "feasibility": res["feasibility"],
                "impact": res["impact"],
                "critique": res["critique"],
                "iteration": res["iteration"],
                "messages": [message_to_dict(message) for message in res["messages"]],
            },
        )

    def __parse_title(self, state: HypgenState) -> str:
        title_match = re.search(r"Title:.*“(.+?)”", state["hypothesis"])
        if title_match:
            return title_match.group(1)
        return f"Hypothesis for {state['subgraph']}"

    def __parse_statement(self, state: HypgenState) -> str:
        statement_match = re.search(
            r"Hypothesis Statement:(.+?)$", state["hypothesis"], re.DOTALL
        )
        if statement_match:
            return statement_match.group(1)
        return state["hypothesis"]

    def __str__(self) -> str:
        return "HypeGen Generator"

    def to_json(self) -> dict[str, Any]:
        return {"type": "HypothesisGenerator"}
