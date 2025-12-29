from fastmcp import FastMCP
from ddgs import DDGS

mcp = FastMCP("MCP with websearch")


@mcp.tool
def websearch(search_query: str, max_results: int = 5) -> list[dict]:
    """Searches the web for relevant text information and returns a list as dictionary"""
    web_results = DDGS().text(
        query=search_query,
        safesearch="moderate",
        backend="auto",
        max_results=max_results,
    )
    return web_results


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="localhost", port=8080)
