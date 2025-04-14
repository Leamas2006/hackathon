import logging
import os
from typing import Any, Dict, List
import json

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

from ard.hypothesis import Hypothesis, HypothesisGeneratorProtocol
from ard.subgraph import Subgraph

logger = logging.getLogger("hypothesis_generator")


class MultiAgentHypothesisGenerator(HypothesisGeneratorProtocol):
    """A multi-agent system for generating scientific hypotheses from subgraphs."""

    def __init__(self):
        """Initialize the hypothesis generator with required agents."""
        self.model_name = os.environ.get("OPENAI_MODEL", "gpt-4")
        self.temperature = float(os.environ.get("OPENAI_TEMPERATURE", 0.7))
        self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("Missing OPENAI_API_KEY. Please set it in your environment variables.")

        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.api_key
        )

        # Initialize conversation history for logging purposes
        self.conversation_history = []

    def run(self, subgraph: Subgraph) -> Hypothesis:
        """Generate a hypothesis from a subgraph using multiple collaborative agents.

        Args:
            subgraph: The source subgraph to analyze

        Returns:
            A fully formed Hypothesis object
        """
        # Log the start of hypothesis generation
        logger.info(
            f"Starting hypothesis generation for subgraph: {subgraph.subgraph_id}")

        # Extract key information from subgraph
        context = subgraph.context or "No additional context provided."
        path = subgraph.to_cypher_string(full_graph=False)
        full_graph = subgraph.to_cypher_string(full_graph=True)

        # Step 1: Ontology Agent - Define concepts and relationships
        logger.info(
            "Step 1: Ontology Agent analyzing concepts and relationships")
        ontology_analysis = self._run_ontology_agent(subgraph, path, context)
        self.conversation_history.append(
            {"agent": "ontology", "content": ontology_analysis})

        # Step 2: Research Agent - Generate initial hypothesis
        logger.info("Step 2: Research Agent generating initial hypothesis")
        initial_hypothesis = self._run_research_agent(
            subgraph, path, ontology_analysis, context)
        self.conversation_history.append(
            {"agent": "research", "content": initial_hypothesis})

        # Step 3: Critic Agent - Evaluate and provide feedback
        logger.info("Step 3: Critic Agent evaluating hypothesis")
        critique = self._run_critic_agent(
            initial_hypothesis, ontology_analysis, full_graph)
        self.conversation_history.append(
            {"agent": "critic", "content": critique})

        # Step 4: Refinement Agent - Refine hypothesis based on critique
        logger.info("Step 4: Refinement Agent improving hypothesis")
        refined_hypothesis = self._run_refinement_agent(
            initial_hypothesis, critique, ontology_analysis, path)
        self.conversation_history.append(
            {"agent": "refinement", "content": refined_hypothesis})

        # Step 5: Literature Agent - Find supporting references
        logger.info("Step 5: Literature Agent finding references")
        references = self._run_literature_agent(refined_hypothesis, subgraph)
        self.conversation_history.append(
            {"agent": "literature", "content": str(references)})

        # Step 6: Final Synthesis Agent - Create the final hypothesis with all components
        logger.info("Step 6: Synthesis Agent creating final hypothesis")
        title, statement = self._run_synthesis_agent(
            refined_hypothesis, references, ontology_analysis)

        # Create and return the Hypothesis object
        return Hypothesis(
            title=title,
            statement=statement,
            source=subgraph,
            method=self,
            references=references,
            metadata={
                "conversation_history": self.conversation_history,
                "ontology_analysis": ontology_analysis,
                "initial_hypothesis": initial_hypothesis,
                "critique": critique,
                "refined_hypothesis": refined_hypothesis
            }
        )

    def _run_ontology_agent(self, subgraph: Subgraph, path: str, context: str) -> str:
        """Agent that analyzes and defines the key concepts in the subgraph."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an Ontology Agent specializing in scientific concept definition and relationship analysis. 
            Your job is to carefully define each concept in the provided subgraph and explain the relationships between them.
            Analyze the provided knowledge graph path and identify the key scientific concepts and their relationships.
            For each concept, provide:
            1. A clear scientific definition
            2. Its role in the overall graph
            3. Key properties and characteristics
            
            For each relationship, explain:
            1. The scientific basis for the connection
            2. The directionality (how one concept influences another)
            3. Potential mechanisms of interaction
            
            Be precise, scientifically accurate, and comprehensive.
            """),
            HumanMessage(content=f"""
            Please analyze the following knowledge graph path and provide detailed definitions and relationship analyses:
            
            PATH:
            {path}
            
            ADDITIONAL CONTEXT:
            {context}
            
            Organize your response in a clear, structured format with sections for each concept and relationship.
            """)
        ])

        response = self.llm(prompt.messages)
        return response.content

    def _run_research_agent(self, subgraph: Subgraph, path: str, ontology_analysis: str, context: str) -> str:
        """Agent that generates the initial hypothesis based on the subgraph."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Research Agent with expertise in generating novel scientific hypotheses. 
            Your task is to examine the knowledge graph path and ontological analysis, then formulate an initial scientific hypothesis.
            
            Your hypothesis should:
            1. Explain the mechanism of connection between the start and end nodes
            2. Be specific, testable, and falsifiable
            3. Incorporate multiple concepts from the knowledge graph
            4. Suggest a novel perspective or insight not immediately obvious from the graph
            5. Consider potential causal relationships and mechanisms
            
            Be creative but scientifically grounded. Your hypothesis should represent a novel contribution to scientific understanding.
            """),
            HumanMessage(content=f"""
            Based on the following knowledge graph path and ontological analysis, generate a novel scientific hypothesis:
            
            PATH:
            {path}
            
            ONTOLOGICAL ANALYSIS:
            {ontology_analysis}
            
            ADDITIONAL CONTEXT:
            {context}
            
            Start node: {subgraph.start_node}
            End node: {subgraph.end_node}
            
            Formulate a detailed hypothesis that explains the relationship between these concepts in a novel way.
            """)
        ])

        response = self.llm(prompt.messages)
        return response.content

    def _run_critic_agent(self, hypothesis: str, ontology_analysis: str, full_graph: str) -> str:
        """Agent that critically evaluates the hypothesis for scientific merit and gaps."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Critic Agent with expertise in evaluating scientific hypotheses. 
            Your task is to carefully analyze the proposed hypothesis and provide constructive criticism.
            
            Evaluate the hypothesis based on:
            1. Scientific validity and plausibility
            2. Alignment with the knowledge graph concepts and relationships
            3. Novelty and potential scientific impact
            4. Logical coherence and causal mechanisms
            5. Testability and falsifiability
            6. Potential gaps, weaknesses, or overlooked aspects
            
            Provide specific suggestions for improvement, additional considerations, and alternative perspectives.
            Be rigorous but constructive in your criticism.
            """),
            HumanMessage(content=f"""
            Please critically evaluate the following scientific hypothesis:
            
            HYPOTHESIS:
            {hypothesis}
            
            ONTOLOGICAL ANALYSIS:
            {ontology_analysis}
            
            FULL KNOWLEDGE GRAPH:
            {full_graph}
            
            Provide a detailed critique with specific points for improvement.
            """)
        ])

        response = self.llm(prompt.messages)
        return response.content

    def _run_refinement_agent(self, hypothesis: str, critique: str, ontology_analysis: str, path: str) -> str:
        """Agent that refines the hypothesis based on the critique."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Refinement Agent specializing in improving scientific hypotheses. 
            Your task is to refine an initial hypothesis based on critical feedback and ontological understanding.
            
            In your refinement:
            1. Address all valid criticisms and suggestions
            2. Strengthen the scientific reasoning and causal mechanisms
            3. Improve specificity and clarity
            4. Enhance testability and falsifiability
            5. Ensure comprehensive integration of knowledge graph concepts
            6. Maintain or enhance the novelty of the insight
            
            Provide a completely rewritten, improved hypothesis that addresses all the feedback.
            """),
            HumanMessage(content=f"""
            Please refine the following scientific hypothesis based on the critique and ontological analysis:
            
            INITIAL HYPOTHESIS:
            {hypothesis}
            
            CRITIQUE:
            {critique}
            
            ONTOLOGICAL ANALYSIS:
            {ontology_analysis}
            
            PATH:
            {path}
            
            Provide a comprehensive, refined hypothesis that addresses the critique while maintaining scientific novelty.
            """)
        ])

        response = self.llm(prompt.messages)
        return response.content

    def _run_literature_agent(self, refined_hypothesis: str, subgraph: Subgraph) -> List[str]:
        """Agent that generates relevant scientific references to support the hypothesis."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Literature Agent with expertise in scientific publication knowledge. 
            Your task is to generate plausible and relevant scientific references that would support the given hypothesis.
            
            For each reference:
            1. Create a title that accurately reflects research related to the hypothesis
            2. Use real author surnames but fictional initials
            3. Include a plausible journal name in the field
            4. Use a realistic publication year (within the last 15 years)
            5. Make sure the reference content genuinely supports some aspect of the hypothesis
            
            Format each reference in standard academic citation format.
            Generate 3-5 high-quality, plausible references that collectively support different aspects of the hypothesis.
            """),
            HumanMessage(content=f"""
            Please generate relevant scientific references to support the following hypothesis:
            
            REFINED HYPOTHESIS:
            {refined_hypothesis}
            
            KEY CONCEPTS:
            {', '.join([subgraph.start_node, subgraph.end_node] + [node for node in subgraph.path_nodes if node not in [subgraph.start_node, subgraph.end_node]])}
            
            Generate 3-5 plausible references in standard academic citation format.
            """)
        ])

        response = self.llm(prompt.messages)

        # Extract references as a list
        references = []
        for line in response.content.strip().split("\n"):
            if line.strip() and not line.startswith("#") and not line.startswith("-"):
                references.append(line.strip())

        return references

    def _run_synthesis_agent(self, refined_hypothesis: str, references: List[str], ontology_analysis: str) -> tuple[str, str]:
        """Agent that creates the final hypothesis with title and statement."""
        refs_text = "\n".join(references)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Synthesis Agent specializing in creating clear, compelling scientific hypotheses. 
            Your task is to create a final hypothesis with a concise title and detailed statement.
            
            For the title:
            1. Be concise (10-15 words)
            2. Clearly indicate the main relationship or mechanism
            3. Use scientific terminology appropriately
            4. Be attention-grabbing but scientifically accurate
            
            For the statement:
            1. Clearly articulate the complete hypothesis in 2-4 sentences
            2. Be specific about mechanisms and relationships
            3. Ensure testability and falsifiability
            4. Include the key concepts and their interactions
            5. Be scientifically rigorous and precise
            
            Format your response as:
            TITLE: [Your concise title]
            STATEMENT: [Your detailed hypothesis statement]
            """),
            HumanMessage(content=f"""
            Please create a final hypothesis with title and statement based on the following:
            
            REFINED HYPOTHESIS:
            {refined_hypothesis}
            
            SUPPORTING REFERENCES:
            {refs_text}
            
            ONTOLOGICAL ANALYSIS:
            {ontology_analysis}
            
            Create a concise title and detailed hypothesis statement that represents a novel scientific contribution.
            """)
        ])

        response = self.llm(prompt.messages)
        content = response.content

        # Extract title and statement
        title_match = content.split("TITLE:")[1].split("STATEMENT:")[
            0].strip() if "TITLE:" in content else ""
        statement_match = content.split("STATEMENT:")[1].strip(
        ) if "STATEMENT:" in content else content

        return title_match, statement_match

    def __str__(self) -> str:
        """String representation of the generator."""
        return "MultiAgentHypothesisGenerator"

    def to_json(self) -> dict[str, Any]:
        """JSON serialization for the generator."""
        return {
            "name": str(self),
            "type": "multi_agent_system",
            "model": self.model_name,
            "temperature": self.temperature
        }
