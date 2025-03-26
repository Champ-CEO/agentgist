from dataclasses import dataclass

from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint
from loguru import logger

from agentgist.agents import (
    analyze_post,
    fetch_subreddit_posts,
    filter_posts,
    request_filter_query,
    write_report,
)
from agentgist.config import Config
from agentgist.data import Report
from agentgist.models import create_llm
from agentgist.groq_strategies import DynamicComplexityRouter


@dataclass
class ReportWorkflowInput:
    subreddit: str
    max_posts: int


@entrypoint(checkpointer=MemorySaver())
def write_report_workflow(workflow_input: ReportWorkflowInput) -> Report:
    logger.debug(f"Fetching {workflow_input.max_posts} posts from {workflow_input.subreddit}")

    posts = fetch_subreddit_posts(
        workflow_input.subreddit, max_posts=workflow_input.max_posts
    ).result()
    logger.debug(f"Fetched posts: {[p.model_dump() for p in posts]}")

    query = request_filter_query(posts).result()
    logger.debug(f"User query: {query}")

    # Pass query to create_llm for dynamic model selection based on complexity
    complexity = DynamicComplexityRouter.assess_complexity(query)
    logger.debug(f"Query complexity assessed as: {complexity}")

    # Create LLM with dynamic routing based on query complexity
    llm = create_llm(Config.Model.DEFAULT, query=query)
    selected_posts = filter_posts(llm, query, posts).result()
    logger.debug(f"Selected posts: {[p.model_dump() for p in selected_posts]}")

    for post in selected_posts:
        # Analyze each post with dynamic routing - post analysis needs accuracy
        post_specific_query = f"Analyze post: {post.title}. {query}"
        post_llm = create_llm(Config.Model.DEFAULT, query=post_specific_query)
        post.analysis = analyze_post(post_llm, query, post).result()
        logger.debug(post.analysis.model_dump())

    # For report writing, explicitly use the report writer model with the query
    # Report writing is inherently complex, but we still pass the query for token optimization
    report_llm = create_llm(Config.Model.REPORT_WRITER, query=query)
    report = write_report(report_llm, query, selected_posts).result()
    logger.debug(f"Report: {report.model_dump()}")

    report.references = selected_posts
    return report
