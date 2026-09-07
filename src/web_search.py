import os

from tavily import TavilyClient


def web_search(
    query,
    max_results=5
):

    api_key = os.getenv(
        "TAVILY_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "TAVILY_API_KEY is not configured."
        )

    client = TavilyClient(
        api_key=api_key
    )

    response = client.search(
        query=query,
        max_results=max_results
    )

    return response