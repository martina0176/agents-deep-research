"""Agent used to perform vector database searches."""

from ...tools.vector_db_search import create_vector_search_tool
from ...llm_config import LLMConfig, model_supports_structured_output
from . import ToolAgentOutput
from ..baseclass import ResearchAgent
from ..utils.parse_output import create_type_parser

INSTRUCTIONS = f"""You are a research assistant that uses a vector database to answer questions.

Use the vector_db_search tool to retrieve relevant information from the database
and return a short summary referencing the sources when possible.

Only output JSON. Follow the JSON schema below. Do not output anything else:
{ToolAgentOutput.model_json_schema()}"""


def init_vector_search_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.fast_model
    vector_tool = create_vector_search_tool(config)

    return ResearchAgent(
        name="VectorSearchAgent",
        instructions=INSTRUCTIONS,
        tools=[vector_tool],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_structured_output(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_structured_output(selected_model) else None
    )
