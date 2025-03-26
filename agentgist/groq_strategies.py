"""
Groq integration strategies for AgentGist.

This module implements Groq's recommended Accuracy-Preserving Strategies and
Cost-Saving Components for optimal LLM usage.

References:
- Dynamic Complexity Routing: Routes tasks to appropriate models based on complexity
- Token Optimization Pipeline: Reduces token usage through preprocessing and prompt optimization
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from pydantic import Field, PrivateAttr

from agentgist.config import Config, ModelConfig


class TaskComplexity(str, Enum):
    """Enum for task complexity levels used in Dynamic Complexity Routing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TokenOptimizer:
    """Implements token optimization strategies to reduce API costs."""

    @staticmethod
    def truncate_context(text: str, max_tokens: int = 8000) -> str:
        """Truncate text to approximately max_tokens."""
        # Simple approximation: ~4 chars per token
        char_limit = max_tokens * 4
        if len(text) <= char_limit:
            return text

        # Keep introduction and conclusion, truncate middle
        intro_size = char_limit // 4
        conclusion_size = char_limit // 4
        middle_size = char_limit - (intro_size + conclusion_size)

        intro = text[:intro_size]
        conclusion = text[-conclusion_size:]
        middle_start = (len(text) - middle_size) // 2
        middle = text[middle_start : middle_start + middle_size]

        return f"{intro}...[content truncated]...{middle}...[content truncated]...{conclusion}"

    @staticmethod
    def optimize_prompt(prompt: str) -> str:
        """Optimize prompts to be more token-efficient."""
        # Remove redundant whitespace
        optimized = " ".join(prompt.split())

        # Replace common verbose phrases with shorter equivalents
        replacements = {
            "in order to": "to",
            "for the purpose of": "to",
            "a large number of": "many",
            "a significant amount of": "much",
            "at this point in time": "now",
            "due to the fact that": "because",
            "in the event that": "if",
        }

        for verbose, concise in replacements.items():
            optimized = optimized.replace(verbose, concise)

        return optimized


class DynamicComplexityRouter:
    """Routes tasks to appropriate models based on complexity assessment."""

    complexity_keywords = {
        TaskComplexity.HIGH: [
            "report",
            "analyze",
            "synthesize",
            "evaluate",
            "compare",
            "critique",
            "recommend",
            "develop",
            "create",
            "design",
            "summarize",
            "comprehensive",
            "detailed",
            "nuanced",
        ],
        TaskComplexity.MEDIUM: [
            "explain",
            "describe",
            "outline",
            "list",
            "identify",
            "find",
            "filter",
            "search",
            "calculate",
            "process",
        ],
        TaskComplexity.LOW: [
            "check",
            "verify",
            "confirm",
            "is",
            "simple",
            "basic",
            "quick",
            "brief",
            "yes/no",
        ],
    }

    @classmethod
    def assess_complexity(cls, query: str) -> TaskComplexity:
        """Determine the complexity level of a given query."""
        query = query.lower()

        # Count keyword matches for each complexity level
        scores = {complexity: 0 for complexity in TaskComplexity}
        for complexity, keywords in cls.complexity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query:
                    scores[complexity] += 1

        # Handle ties by preferring lower complexity (cost saving)
        if scores[TaskComplexity.HIGH] > scores[TaskComplexity.MEDIUM]:
            return TaskComplexity.HIGH
        elif scores[TaskComplexity.MEDIUM] > scores[TaskComplexity.LOW]:
            return TaskComplexity.MEDIUM
        return TaskComplexity.LOW

    @classmethod
    def get_model_for_task(cls, query: str) -> ModelConfig:
        """Return the appropriate model config based on query complexity."""
        complexity = cls.assess_complexity(query)

        if complexity == TaskComplexity.HIGH:
            return Config.Model.REPORT_WRITER
        return Config.Model.DEFAULT


class EnhancedGroqChat(BaseChatModel):
    """
    A properly implemented Groq chat model with optimization features.
    """

    model: str = Field(description="The name of the Groq model to use")
    temperature: float = Field(default=0.0, description="The temperature to use for sampling")
    optimize_tokens: bool = Field(default=True, description="Whether to optimize tokens")

    _chat_model: Any = PrivateAttr()
    _token_optimizer: TokenOptimizer = PrivateAttr()

    def __init__(self, **kwargs):
        """Initialize the model with Groq-specific parameters."""
        super().__init__(**kwargs)

        # Initialize private attributes
        self._chat_model = ChatGroq(model=self.model, temperature=self.temperature)
        self._token_optimizer = TokenOptimizer()

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate method required by BaseChatModel."""
        if self.optimize_tokens:
            messages = self._optimize_messages(messages)

        return self._chat_model._generate(messages, stop, run_manager, **kwargs)

    def _optimize_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Apply token optimization strategies to messages."""
        optimized_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                # Optimize human prompts
                optimized_content = self._token_optimizer.optimize_prompt(message.content)
                optimized_messages.append(HumanMessage(content=optimized_content))
            elif isinstance(message, SystemMessage) and len(message.content) > 1000:
                # Truncate very long system messages
                truncated_content = self._token_optimizer.truncate_context(
                    message.content, max_tokens=2000
                )
                optimized_messages.append(SystemMessage(content=truncated_content))
            else:
                # Keep other messages as-is
                optimized_messages.append(message)

        return optimized_messages

    def bind_tools(self, tools: List[BaseTool], **kwargs: Any) -> BaseChatModel:
        """
        Bind tools to the chat model for function calling.
        """
        bound_chat_model = self._chat_model.bind_tools(tools, **kwargs)

        # Create a new instance with the bound chat model
        new_instance = self.__class__(
            model=self.model, temperature=self.temperature, optimize_tokens=self.optimize_tokens
        )
        new_instance._chat_model = bound_chat_model

        return new_instance

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> Any:
        """Apply token optimization before invoking the model."""
        if self.optimize_tokens:
            messages = self._optimize_messages(messages)

        return self._chat_model.invoke(messages, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "enhanced-groq"


def create_optimized_chat(
    model_config: ModelConfig, query: Optional[str] = None, optimize_tokens: bool = True
) -> BaseChatModel:
    """
    Create an optimized chat model with dynamic routing if query is provided.

    Args:
        model_config: The default model configuration to use
        query: Optional query text to dynamically route to appropriate model
        optimize_tokens: Whether to enable token optimization

    Returns:
        An optimized chat model for the task
    """
    # If query is provided, dynamically select model based on complexity
    if query:
        model_config = DynamicComplexityRouter.get_model_for_task(query)

    # Create the optimized Groq chat model
    return EnhancedGroqChat(
        model=model_config.name,
        temperature=model_config.temperature,
        optimize_tokens=optimize_tokens,
    )
