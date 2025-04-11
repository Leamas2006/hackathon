import time
from pathlib import Path

import click
from loguru import logger
from openai import OpenAI

from ard.data.dataset import Dataset
from ard.knowledge_graph import KnowledgeGraph
from ard.knowledge_graph.node_merger.embedding_based import EmbeddingBasedNodeMerger
from ard.subgraph.subgraph import Subgraph
from ard.subgraph.subgraph_generator import (
    RandomizedEmbeddingPathGenerator,
)
from ard.subgraph.subgraph_generator.embedding import EmbeddingPathGenerator
from ard.subgraph.subgraph_generator.llm_walk import LLMWalkGenerator
from ard.subgraph.subgraph_generator.random_walk import (
    SingleNodeRandomWalkGenerator,
)
from ard.subgraph.subgraph_generator.shortest_path import ShortestPathGenerator
from ard.utils.embedder import Embedder

client = OpenAI()


def get_llm(model: str = "gpt-4o-mini"):
    if model == "small":
        model = "gpt-4o-mini"
    elif model == "large":
        model = "gpt-4o"
    elif model == "reasoning":
        model = "o3-mini"

    def llm(prompt: str):
        response = client.responses.create(
            model=model,
            input=prompt,
        )
        return response.output_text

    return llm


def log_timing(operation_name, start_time):
    """Log the time taken for an operation in a consistent format."""
    elapsed = time.time() - start_time
    logger.info(f"⏱️ {operation_name} completed in {elapsed:.2f} seconds")


def log_section(section_name):
    """Create a visual separator for a new section in logs."""
    logger.info(f"\n{'=' * 50}")
    logger.info(f"📌 {section_name}")
    logger.info(f"{'=' * 50}")


@click.group()
def cli():
    """ARD - Knowledge Graph and Subgraph Pipeline Tool."""
    pass


@cli.command("graph")
@click.option(
    "--data-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Path to the data directory. If not provided, uses the default example data.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="knowledge_graph.pkl",
    help="Path to save the knowledge graph (default: knowledge_graph.pkl)",
)
@click.option(
    "--max-items",
    type=int,
    default=10,
    help="Maximum number of items to process (default: 10)",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.85,
    help="Similarity threshold for merging nodes (default: 0.85)",
)
def create_graph(data_path, output, max_items, similarity_threshold):
    """Create a knowledge graph from data."""
    log_section("KNOWLEDGE GRAPH CREATION")

    # Determine data directory
    if data_path:
        data_dir = Path(data_path)
        logger.info(f"📂 Using provided data directory: {data_dir}")
    else:
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        logger.info(f"📂 Using default example data directory: {data_dir}")

    # Create a Dataset from the directory
    logger.info("🔄 Creating dataset from directory...")
    dataset = Dataset.from_local(data_dir)

    # Create a KnowledgeGraph from triplets
    log_section("BUILDING GRAPH")
    logger.info(f"🔄 Building knowledge graph (max_items={max_items})...")
    start_time = time.time()
    kg = KnowledgeGraph.from_dataset(dataset, max_items=max_items)
    log_timing("Knowledge graph creation", start_time)
    logger.debug("Knowledge Graph details:")
    logger.debug(kg)

    # Merge similar nodes using the EmbeddingBasedNodeMerger
    log_section("MERGING SIMILAR NODES")
    logger.info(f"🔄 Merging similar nodes (threshold={similarity_threshold})...")
    merger = EmbeddingBasedNodeMerger(
        embedding_model_name="all-MiniLM-L6-v2",
        similarity_threshold=similarity_threshold,
    )

    start_time = time.time()
    kg.merge_similar_nodes(merger)
    log_timing("Node merging", start_time)
    logger.info(f"✅ Graph after merging: {kg}")
    logger.debug("Merged Graph details:")
    logger.debug(kg)

    # Save the KnowledgeGraph
    log_section("SAVING GRAPH")
    logger.info(f"💾 Saving knowledge graph to {output}...")
    start_time = time.time()
    kg.save_to_file(output)
    log_timing("Graph saving", start_time)
    logger.info(f"✅ Knowledge graph successfully saved to {output}")


@cli.command("subgraph")
@click.option(
    "--graph-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the knowledge graph file. If not provided, uses the default example data.",
)
@click.option(
    "--embedder-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the embedder file. If not provided, it computes the embeddings on the fly.",
)
@click.option(
    "--num-subgraphs",
    type=int,
    default=10,
    help="Number of subgraphs to generate.",
)
@click.option(
    "--max-nodes",
    type=int,
    default=20,
    help="Maximum number of nodes to include in the subgraph.",
)
@click.option(
    "--max-steps",
    type=int,
    default=10,
    help="Maximum number of steps to take in the random walk.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="output",
    help="Output directory for saving subgraphs.",
)
@click.option(
    "--method",
    type=click.Choice(
        [
            "random_walk",
            "llm_walk",
            "embedding_path",
            "randomized_embedding_path",
            "shortest_path",
        ]
    ),
    default="random_walk",
    help="Method to use for subgraph extraction (default: random_walk).",
)
@click.option(
    "--min-score",
    type=float,
    default=4,
    help="Minimum score for a subgraph to be considered valid (default: 4).",
)
@click.option(
    "--neighbor-probability",
    type=float,
    default=0.5,
    help="Probability of selecting a neighbor node (default: 0.5).",
)
@click.option(
    "--llm",
    type=str,
    default="small",
    help="LLM to use for subgraph generation.",
)
def extract_subgraph(
    graph_path,
    embedder_path,
    num_subgraphs,
    max_nodes,
    max_steps,
    output_dir,
    method,
    min_score,
    neighbor_probability,
    llm,
):
    """Extract subgraphs from a knowledge graph."""
    log_section("SUBGRAPH EXTRACTION")
    logger.info("📊 Configuration:")
    logger.info(f"   Method: {method}")
    logger.info(f"   Number of subgraphs: {num_subgraphs}")
    logger.info(f"   Max nodes: {max_nodes}")
    logger.info(f"   Max steps: {max_steps}")
    logger.info(f"   Minimum score: {min_score}")
    logger.info(f"   Neighbor probability: {neighbor_probability}")
    logger.info(f"   LLM: {llm}")

    # Determine graph path
    if graph_path:
        graph_path = Path(graph_path)
        logger.info(f"📂 Using provided knowledge graph: {graph_path}")
    else:
        graph_path = (
            Path(__file__).parent.parent.parent.parent / "data" / "knowledge_graph.pkl"
        )
        logger.info(f"📂 Using default example knowledge graph: {graph_path}")

    # Initialize embedder if provided
    embedder = Embedder()
    if embedder_path:
        logger.info(f"🔄 Loading embedder from {embedder_path}")
        embedder.load_from_file(embedder_path)
    else:
        logger.info("ℹ️ No embedder provided, will compute embeddings on the fly")

    # Load knowledge graph
    logger.info("🔄 Loading knowledge graph...")
    start_time = time.time()
    kg = KnowledgeGraph.load_from_file(graph_path)
    log_timing("Knowledge graph loading", start_time)
    logger.info("✅ Loaded graph successfully")
    logger.debug("Knowledge Graph details:")
    logger.debug(kg)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_path}")

    # Generate subgraphs
    for i in range(num_subgraphs):
        log_section(f"GENERATING SUBGRAPH {i + 1}/{num_subgraphs}")
        subgraph = None
        attempt = 0

        while True:
            attempt += 1
            try:
                logger.info(
                    f"🔄 Attempt {attempt}: Generating subgraph with method '{method}'"
                )
                start_time = time.time()
                if method == "random_walk":
                    start_node = kg.get_random_node()
                    logger.info(f"   Starting from node: {start_node}")
                    subgraph = Subgraph.from_one_node(
                        kg,
                        start_node,
                        method=SingleNodeRandomWalkGenerator(max_steps=max_steps),
                        max_nodes=max_nodes,
                        neighbor_probability=neighbor_probability,
                    )
                elif method == "llm_walk":
                    start_node = kg.get_random_node()
                    logger.info(f"   Starting from node: {start_node}")
                    subgraph = Subgraph.from_one_node(
                        kg,
                        start_node,
                        method=LLMWalkGenerator(max_steps=max_steps, llm="small"),
                        max_nodes=max_nodes,
                        neighbor_probability=neighbor_probability,
                    )
                elif method == "embedding_path":
                    start_node = kg.get_random_node()
                    end_node = kg.get_random_node()
                    logger.info(f"   Path from: {start_node} to {end_node}")
                    subgraph = Subgraph.from_two_nodes(
                        kg,
                        start_node,
                        end_node,
                        method=EmbeddingPathGenerator(embedder=embedder),
                        max_nodes=max_nodes,
                        neighbor_probability=neighbor_probability,
                    )
                elif method == "randomized_embedding_path":
                    start_node = kg.get_random_node()
                    end_node = kg.get_random_node()
                    logger.info(f"   Path from: {start_node} to {end_node}")
                    subgraph = Subgraph.from_two_nodes(
                        kg,
                        start_node,
                        end_node,
                        method=RandomizedEmbeddingPathGenerator(embedder=embedder),
                        max_nodes=max_nodes,
                        neighbor_probability=neighbor_probability,
                    )
                elif method == "shortest_path":
                    start_node = kg.get_random_node()
                    end_node = kg.get_random_node()
                    logger.info(f"   Path from: {start_node} to {end_node}")
                    subgraph = Subgraph.from_two_nodes(
                        kg,
                        start_node,
                        end_node,
                        method=ShortestPathGenerator(),
                        max_nodes=max_nodes,
                        neighbor_probability=neighbor_probability,
                    )
                else:
                    raise ValueError(f"Invalid method: {method}")
                log_timing("Subgraph generation", start_time)

                logger.info("🔄 Scoring subgraph path...")
                start_time = time.time()
                subgraph.score_path(get_llm(llm))
                log_timing("Path scoring", start_time)

                score_info = f"Score: {subgraph._path_score:.2f}"
                if subgraph._path_score < min_score:
                    logger.warning(
                        f"⚠️ {score_info} - Below minimum threshold of {min_score}"
                    )
                    logger.debug(f"Subgraph details: {subgraph}")
                    logger.debug(f"Justification: {subgraph._path_score_justification}")
                    logger.warning("🔄 Retrying...")
                    continue
                else:
                    logger.info(f"✅ {score_info} - Above minimum threshold")
                    logger.info("🔄 Contextualizing subgraph...")
                    start_time = time.time()
                    subgraph.contextualize(llm=get_llm(llm))
                    log_timing("Contextualization", start_time)
                    break
            except Exception as e:
                logger.error(f"❌ Error in attempt {attempt}: {e}")
                if attempt >= 3:
                    logger.warning(
                        f"⚠️ Failed after {attempt} attempts, moving to next subgraph"
                    )
                    break
                logger.info("🔄 Retrying...")

        # Save the subgraph
        if subgraph:
            subdir = output_path

            subdir.mkdir(parents=True, exist_ok=True)
            output_name = f"{subgraph.start_node}_{subgraph.end_node}".replace("/", "_")
            output_file = subdir / output_name
            subgraph.save_to_file(output_file.with_suffix(".subgraph.json"))
            logger.info(f"✅ Saved to {output_file}")
        else:
            logger.warning(f"⚠️ No valid subgraph generated for iteration {i + 1}")

    log_section("PROCESS COMPLETED")
    logger.info(f"✅ Generated {num_subgraphs} subgraphs")


if __name__ == "__main__":
    cli()
