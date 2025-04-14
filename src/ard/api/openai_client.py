import openai

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        openai.api_key = "sk-proj-aXvv31x1jAZb-onjHmd7lxy_eHGeYiAFcecJ8YD8qr_iOlw-QN2-o0qdNvMzGJhsmOcy2KBx|WT3B|bkFJkW6NRhd1ENо_olZvSXHAfJgi8J_vY1о2vTler0zFZ5XWBv6QUWAHо1Lbu0iDhq5wMqqPgUSggА"

    def generate_hypothesis(self, graph_nodes, graph_edges, system_prompt=None):
        """
        Generate a hypothesis based on nodes and edges of a graph.
        """
        system_prompt = system_prompt or (
            "You are a scientific research assistant. "
            "Based on the given graph structure, generate a plausible scientific hypothesis "
            "connecting the concepts involved."
        )

        node_list = ", ".join(graph_nodes)
        edge_descriptions = "\n".join([
            f"{e['source']} —[{e.get('relation', 'related to')}]→ {e['target']}"
            for e in graph_edges
        ])

        user_prompt = (
            f"The graph contains the following nodes:\n{node_list}\n\n"
            f"And the following edges:\n{edge_descriptions}\n\n"
            f"Generate a hypothesis based on this structure."
        )

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        return response["choices"][0]["message"]["content"].strip()
