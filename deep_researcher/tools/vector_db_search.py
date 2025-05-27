import json
import os
from typing import List, Dict, Union
from agents import function_tool
from agents.mcp.server import MCPServerSse
from mcp.types import TextContent
from dotenv import load_dotenv

load_dotenv()

VECTOR_SEARCH_SERVER_URL = os.getenv("VECTOR_SEARCH_SERVER_URL")
VECTOR_SEARCH_TOOL_NAME = os.getenv("VECTOR_SEARCH_TOOL_NAME", "search")


def create_vector_search_tool(_config) -> function_tool:
    """Create a tool that queries a vector database via an MCP SSE server."""

    @function_tool
    async def vector_db_search(query: str) -> Union[List[Dict[str, str]], str]:
        """Search a vector database using a remote server.

        Args:
            query: The search query.

        Returns:
            Either a list of result objects with ``content``, ``source``, ``title``
            and ``subtitle`` fields, or a plain error message.
        """
        if not VECTOR_SEARCH_SERVER_URL:
            return "VECTOR_SEARCH_SERVER_URL not set"

        server = MCPServerSse({"url": VECTOR_SEARCH_SERVER_URL})
        await server.connect()
        try:
            try:
                result = await server.call_tool(VECTOR_SEARCH_TOOL_NAME, {"query": query})
            except Exception as e:
                return f"Error calling vector search tool: {e}"

            if not hasattr(result, "content"):
                return str(result)

            parts = [c.text for c in result.content if isinstance(c, TextContent)]
            text = " ".join(parts)

            # Attempt to parse the server response as JSON
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return data
            except Exception:
                pass

            return text
        finally:
            await server.cleanup()

    return vector_db_search
