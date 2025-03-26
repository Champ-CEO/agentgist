from functools import lru_cache
from typing import Optional

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel

from agentgist.config import Config, ModelConfig, ModelProvider
from agentgist.groq_strategies import create_optimized_chat


@lru_cache(maxsize=1)
def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Preprocessing.EMBEDDING_MODEL)


def create_llm(model_config: ModelConfig, query: Optional[str] = None) -> BaseChatModel:
    """
    Create a language model optimized for the task.

    Args:
        model_config: The model configuration to use
        query: Optional query to route to appropriate model based on complexity

    Returns:
        A language model optimized for the task
    """
    if model_config.provider == ModelProvider.GROQ:
        # Use enhanced Groq strategies including dynamic routing and token optimization
        return create_optimized_chat(model_config, query=query)
