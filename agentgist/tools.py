from typing import Annotated, List

from langchain_core.documents import Document
from langchain_core.tools import InjectedToolArg, tool
from langchain_core.vectorstores import InMemoryVectorStore

from agentgist.config import Config
from agentgist.models import create_embeddings


@tool
def search_documents(
    query: str, documents: Annotated[List[Document], InjectedToolArg]
) -> List[str]:
    """Search for posts (using their embeddings) based on a given query

    Args:
        query (str): what posts to search for

    Returns:
        List[str]: list of posts that match the query
    """

    vector_store = InMemoryVectorStore.from_documents(documents, create_embeddings()).as_retriever(
        search_kwargs={"k": Config.Preprocessing.FILTER_POST_COUNT}
    )
    return vector_store.invoke(query)
