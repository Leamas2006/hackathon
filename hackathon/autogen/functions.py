"""Functions for Hypegen agents.

This module contains function definitions that are registered with agents for execution.
"""

import os
import random
from pathlib import Path
from typing import Annotated, Union

import requests


def response_to_query_perplexity(
    query: Annotated[
        str,
        """the query for the paper search. The query must consist of relevant keywords separated by +""",
    ],
) -> str:
    """Search for academic papers using the Perplexity API.

    Args:
        query: The search query with keywords separated by +

    Returns:
        The response data from Perplexity API
    """
    url = "https://api.perplexity.ai/chat/completions"

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "system",
                "content": "Search scientific papers for the most relevant papers on the query. Return the top 10 results.",
            },
            {"role": "user", "content": query},
        ],
        "temperature": 0.2,
        "top_p": 0.9,
        "search_domain_filter": None,
        "return_images": False,
        "return_related_questions": False,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "response_format": None,
    }

    headers = {
        "Authorization": f"Bearer {os.getenv('PPLX_API_KEY')}",
        "Content-Type": "application/json",
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    # Check response status
    if response.status_code == 200:
        response_data = response.json()
    else:
        response_data = (
            f"Request failed with status code {response.status_code}: {response.text}"
        )

    return response_data


def response_to_query(
    query: Annotated[
        str,
        """the query for the paper search. The query must consist of relevant keywords separated by +""",
    ],
) -> str:
    """Search for academic papers using the Semantic Scholar API.

    Args:
        query: The search query with keywords separated by +

    Returns:
        The response data from Semantic Scholar API
    """
    # Define the API endpoint URL
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    # More specific query parameter
    query_params = {"query": {query}, "fields": "title,abstract,openAccessPdf,url"}

    # Define headers with API key
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers = {"x-api-key": api_key}
    else:
        headers = {}

    # Send the API request
    response = requests.get(url, params=query_params, headers=headers)

    # Check response status
    if response.status_code == 200:
        response_data = response.json()
        # Process and print the response data as needed
    else:
        response_data = (
            f"Request failed with status code {response.status_code}: {response.text}"
        )

    return response_data


def generate_path(
    keyword_1: Annotated[
        Union[str, None],
        "the first node in the knowledge graph. None for random selection.",
    ],
    keyword_2: Annotated[
        Union[str, None],
        "the second node in the knowledge graph. None for random selection.",
    ],
) -> str:
    """Create a knowledge path between two nodes.

    The function may either take two keywords as the input or randomly assign them
    and then returns a path between these nodes.

    Args:
        keyword_1: The first node in the knowledge graph (None for random selection)
        keyword_2: The second node in the knowledge graph (None for random selection)

    Returns:
        A path containing several concepts (nodes) and relationships between them
    """
    # Import necessary modules
    from ard.knowledge_graph.knowledge_graph import KnowledgeGraph
    from ard.pipelines.subgraph import generate_subgraph

    # Use the examples/hypegen/output directory for storing output
    with Path("./examples/hypegen/output") as output_dir:
        # Load the knowledge graph to find nodes if keywords are provided
        default_graph_path = (
            Path(__file__).parent.parent.parent.parent / "data" / "knowledge_graph.pkl"
        )
        alternate_graph_path = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "knowledge_graph_100.pkl"
        )
        small_graph_path = (
            Path(__file__).parent.parent.parent.parent / "data" / "small_graph.pkl"
        )

        # Try loading the graph files in order of preference
        graph_path = None
        for path in [default_graph_path, alternate_graph_path, small_graph_path]:
            if path.exists():
                graph_path = path
                break

        if not graph_path:
            return "Failed to find a knowledge graph file."

        try:
            kg = KnowledgeGraph.load_from_file(graph_path)
        except Exception as e:
            return f"Failed to load knowledge graph: {str(e)}"

        # If keywords are provided, try to find matching nodes
        start_node = None
        end_node = None

        if keyword_1 or keyword_2:
            all_nodes = list(kg.get_nodes())

            # Find nodes containing the keywords
            if keyword_1:
                matching_nodes = [
                    node for node in all_nodes if keyword_1.lower() in node.lower()
                ]
                if matching_nodes:
                    start_node = random.choice(matching_nodes)

            if keyword_2:
                matching_nodes = [
                    node for node in all_nodes if keyword_2.lower() in node.lower()
                ]
                if matching_nodes:
                    end_node = random.choice(matching_nodes)

            # If we found matching nodes, use shortest_path method
            if start_node and end_node:
                method = "shortest_path"
            elif start_node or end_node:
                # If only one keyword matched, use llm_walk from that node
                method = "llm_walk"
            else:
                # No keywords matched, use default method
                method = "llm_walk"
        else:
            # No keywords provided, use default method
            method = "llm_walk"

        # Generate the subgraph
        subgraph = generate_subgraph(
            graph_path=str(graph_path),
            embedder_path=None,  # Use default embedder or compute on the fly
            max_nodes=10,
            max_steps=5,
            output_dir=output_dir,
            method=method,
            min_score=3,
            neighbor_probability=0.2,
            llm="large",
            max_attempts=3,
        )

        if not subgraph:
            return "Failed to generate a valid path."

        # Use the to_cypher_string method to get the path representation
        cypher_string = subgraph.to_cypher_string()

        # Return the Cypher string as is
        return cypher_string


def rate_novelty_feasibility(
    hypothesis: Annotated[str, "the research hypothesis."],
) -> str:
    """Rate the novelty and feasibility of a research idea against the literature.

    Args:
        hypothesis: The research hypothesis to evaluate

    Returns:
        A rating of novelty and feasibility from 1 to 10
    """
    # Import here to avoid circular imports
    from ard.hypegen.agents import novelty_admin, novelty_assistant

    res = novelty_admin.initiate_chat(
        novelty_assistant,
        clear_history=True,
        silent=False,
        max_turns=10,
        message=f"""Rate the following research hypothesis\n\n{hypothesis}. \n\nCall the function three times at most, but not in parallel. Wait for the results before calling the next function. """,
        summary_method="reflection_with_llm",
        summary_args={
            "summary_prompt": "Return all the results of the analysis as is."
        },
    )

    return res.summary
