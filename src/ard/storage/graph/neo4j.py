from typing import Any, Dict, List, Set, Tuple

from loguru import logger
from neo4j import GraphDatabase

from ard.storage.graph import GraphBackend


class Neo4jBackend(GraphBackend):
    """
    Neo4j backend implementation for the knowledge graph.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        """
        Initialize a new Neo4j backend.

        Args:
            uri (str): The Neo4j database URI
            user (str): The database user
            password (str): The database password
            database (str): The database name
        """
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        self._ensure_constraints()

    @classmethod
    def from_serializable(cls, data: Dict[str, Any], **connection_params):
        """
        Initialize a new Neo4j backend from a serialized dictionary.

        Args:
            data (Dict[str, Any]): Serialized graph data from to_serializable()
            **connection_params: Neo4j connection parameters (uri, user, password, database)

        Returns:
            Neo4jBackend: New backend instance with all data loaded
        """
        # Create Neo4j backend with connection parameters
        backend = cls(**connection_params)

        # Add all nodes with their attributes
        for node_data in data["nodes"]:
            node_id = node_data["id"]
            attrs = node_data.get("attributes", {})
            backend.add_node(node_id, **attrs)

        # Add all edges with their attributes
        for edge_data in data["edges"]:
            source = edge_data["source"]
            target = edge_data["target"]
            attrs = edge_data.get("attributes", {})
            backend.add_edge(source, target, **attrs)

        return backend

    def to_serializable(self) -> Dict[str, Any]:
        """
        Convert the Neo4j graph to a serializable dictionary.

        Returns:
            Dict[str, Any]: A serializable representation of the graph
        """
        # Get all nodes with their attributes
        nodes = []
        for node in self.get_nodes():
            attrs = self.get_node_attrs(node)
            nodes.append({"id": node, "attributes": attrs})

        # Get all edges with their attributes
        edges = []
        for source, target, attrs in self.get_edges():
            edges.append({"source": source, "target": target, "attributes": attrs})

        return {"nodes": nodes, "edges": edges}

    def _ensure_constraints(self) -> None:
        """Ensure required constraints exist in the database."""
        # neo4j does not support label-less node indexes. tbd later when our KG become large
        pass

    def _close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()

    def add_node(self, node: str, **attrs) -> None:
        """Add a node with optional attributes."""
        with self._driver.session(database=self._database) as session:
            props = {k: v for k, v in attrs.items() if v is not None}
            props["name"] = node

            # Create a new node and then look up its name
            result = session.run(
                """
                CREATE (n)
                SET n += $props
                RETURN n.name AS name
                """,
                props=props,
            )
            # Log the new node creation
            name = result.single()["name"]
            logger.debug(f"Created node with name: {name}")

    def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node_id: The name of the node

        Returns:
            bool: True if the node exists, False otherwise
        """
        query = "MATCH (n) WHERE n.name = $name RETURN count(n) AS count"

        with self._driver.session(database=self._database) as session:
            result = session.run(query, name=node_id)
            return result.single()["count"] > 0

    def add_edge(self, source: str, target: str, **attrs) -> None:
        """Add an edge with optional attributes using names."""
        with self._driver.session(database=self._database) as session:
            # Convert attributes to Neo4j format
            props = {k: v for k, v in attrs.items() if v is not None}

            # Use name to find source and target nodes
            query = """
            MATCH (source), (target)
            WHERE source.name = $source AND target.name = $target
            CREATE (source)-[r]->(target)
            SET r += $props
            RETURN r.edge as edge
            """
            result = session.run(query, source=source, target=target, props=props)
            # Log the edge creation
            if result.single():
                logger.debug(f"Created edge from {source} to {target}")
            else:
                logger.warning(f"Failed to create edge from {source} to {target}")

    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (source)-[r]->(target)
                WHERE source.name = $source AND target.name = $target
                RETURN count(r) as count
                """,
                source=source,
                target=target,
            )
            return result.single()["count"] > 0

    def get_node_attrs(self, node: str) -> Dict[str, Any]:
        """Get all attributes of a node by its name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.name = $id
                RETURN properties(n) as props
                """,
                id=node,
            )
            record = result.single()
            if not record:
                return {}

            props = record["props"].copy()

            # If there's no 'sources' list but there are other properties that should be
            # in source metadata, create a sources list
            if (
                "sources" not in props and len(props) > 1
            ):  # More than just the name property
                # Create a normalized structure with sources
                normalized_props = {}

                # Move metadata to sources list
                source_entry = {
                    k: v for k, v in props.items() if k != "name" and k != "sources"
                }
                if source_entry:  # Only add sources if there's actual metadata
                    normalized_props["sources"] = [source_entry]

                return normalized_props

            return props

    def get_edge_attrs(self, source: str, target: str) -> Dict[str, Any]:
        """Get all attributes of an edge using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (source)-[r]->(target)
                WHERE source.name = $source AND target.name = $target
                RETURN properties(r) as props
                """,
                source=source,
                target=target,
            )
            record = result.single()
            if not record:
                return {}

            props = record["props"]

            # Normalize format to match NetworkX backend
            # If 'edge' exists but 'relation' doesn't, rename it
            if "edge" in props and "relation" not in props:
                props["relation"] = props["edge"]

            # Create a sources array if it doesn't exist
            if "sources" not in props:
                # Gather metadata from the edge itself to create a source entry
                source_entry = {k: v for k, v in props.items() if k not in ["relation"]}
                # Include the relation in the source entry
                if "relation" in props:
                    source_entry["relation"] = props["relation"]
                # Remove metadata keys from the top level, leaving only 'relation' and 'sources'
                for k in list(props.keys()):
                    if k != "relation" and k != "sources":
                        props.pop(k)
                # Add the source entry to the sources array
                props["sources"] = [source_entry]

            return props

    def get_nodes(self) -> Set[str]:
        """Get all nodes in the graph using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run("MATCH (n) RETURN n.name AS name")
            return {record["name"] for record in result}

    def get_random_node(self) -> str:
        """Get a random node from the graph using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (n) 
                RETURN n.name AS name
                LIMIT 1
                """
            )
            record = result.single()
            return record["name"] if record else None

    def get_edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all edges in the graph with their attributes using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (source)-[r]->(target)
                RETURN source.name as source, target.name as target, properties(r) AS props
                """
            )
            edges = []
            for record in result:
                source = record["source"]
                target = record["target"]
                props = record[
                    "props"
                ].copy()  # Make a copy to avoid modifying the original

                # Normalize format to match NetworkX backend
                normalized_props = {}

                # Handle relation/edge property
                relation_value = None
                if "edge" in props:
                    relation_value = props.pop("edge")
                    normalized_props["relation"] = relation_value
                elif "relation" in props:
                    relation_value = props["relation"]
                    normalized_props["relation"] = props.pop("relation")

                # Create a sources list if it doesn't exist
                if "sources" not in props:
                    # Use remaining properties as a single source entry
                    source_entry = props.copy()

                    # Add the relation to the source entry
                    if relation_value is not None:
                        source_entry["relation"] = relation_value
                        # Also add as 'edge' to match NetworkX format
                        source_entry["edge"] = relation_value

                    normalized_props["sources"] = [source_entry]
                else:
                    # If sources already exists, use it and ensure each entry has both relation and edge
                    sources = props.pop("sources")
                    for source_entry in sources:
                        if "relation" in source_entry and "edge" not in source_entry:
                            source_entry["edge"] = source_entry["relation"]
                        elif "edge" in source_entry and "relation" not in source_entry:
                            source_entry["relation"] = source_entry["edge"]

                    normalized_props["sources"] = sources

                    # Add any remaining top-level properties to normalized_props
                    for k, v in props.items():
                        normalized_props[k] = v

                edges.append((source, target, normalized_props))

            return edges

    def get_successors(self, node: str) -> List[str]:
        """Get all successor nodes of a node using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (source)-[]->(target)
                WHERE source.name = $id
                RETURN target.name as id
                """,
                id=node,
            )
            return [record["id"] for record in result]

    def get_predecessors(self, node: str) -> List[str]:
        """Get all predecessor nodes of a node using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                MATCH (source)-[]->(target)
                WHERE target.name = $id
                RETURN source.name as id
                """,
                id=node,
            )
            return [record["id"] for record in result]

    def _normalize_edge_data(self, edges_data):
        """Normalize edge data to match NetworkX format, without adding triplet_ids."""
        normalized_edges = []

        for source, target, props in edges_data:
            props = props.copy()  # Work with a copy

            # Create a normalized structure
            normalized = {
                "relation": props.get("relation") or props.get("edge", ""),
                "sources": [],
            }

            # Process the sources list if it exists
            if "sources" in props:
                sources = props.pop("sources")
                for entry in sources:
                    # Ensure both relation and edge exist
                    if "relation" in entry and "edge" not in entry:
                        entry["edge"] = entry["relation"]
                    elif "edge" in entry and "relation" not in entry:
                        entry["relation"] = entry["edge"]

                    normalized["sources"].append(entry)
            else:
                # Create a source entry from the top-level properties
                source_entry = {k: v for k, v in props.items() if k not in ["relation"]}
                source_entry["relation"] = normalized["relation"]
                source_entry["edge"] = normalized["relation"]
                normalized["sources"].append(source_entry)

            normalized_edges.append((source, target, normalized))

        return normalized_edges

    def get_out_edges(self, node: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all outgoing edges of a node with their attributes using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                "MATCH (source)-[r]->(target) WHERE source.name = $id "
                "RETURN source.name as source, target.name as target, properties(r) as props",
                id=node,
            )
            edges = [(r["source"], r["target"], r["props"]) for r in result]
            return self._normalize_edge_data(edges)

    def get_in_edges(self, node: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Get all incoming edges of a node with their attributes using name."""
        with self._driver.session(database=self._database) as session:
            result = session.run(
                "MATCH (source)-[r]->(target) WHERE target.name = $id "
                "RETURN source.name as source, target.name as target, properties(r) as props",
                id=node,
            )
            edges = [(r["source"], r["target"], r["props"].copy()) for r in result]
            return self._normalize_edge_data(edges)

    def remove_node(self, node: str) -> None:
        """Remove a node and all its edges using name."""
        with self._driver.session(database=self._database) as session:
            session.run(
                """
                MATCH (n)
                WHERE n.name = $id
                DETACH DELETE n
                """,
                id=node,
            )

    def number_of_edges(self) -> int:
        """Get the total number of edges in the graph."""
        with self._driver.session(database=self._database) as session:
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            return result.single()["count"]

    def shortest_path(
        self, source: str, target: str, directed: bool = True
    ) -> List[str]:
        """Get the shortest path between two nodes using name."""
        with self._driver.session(database=self._database) as session:
            if directed:
                query = """
                MATCH (source), (target)
                WHERE source.name = $source AND target.name = $target
                MATCH path = shortestPath((source)-[*]->(target))
                RETURN [node.name for node IN nodes(path)] AS path
                """
            else:
                query = """
                MATCH (source), (target)
                WHERE source.name = $source AND target.name = $target
                MATCH path = shortestPath((source)-[*]-(target))
                RETURN [node.name for node IN nodes(path)] AS path
                """
            result = session.run(query, source=source, target=target)
            record = result.single()
            return record["path"] if record else []

    def __len__(self) -> int:
        """Get the total number of nodes in the graph."""
        with self._driver.session(database=self._database) as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            return result.single()["count"]

    def __del__(self):
        """Clean up Neo4j driver connection."""
        self._close()
