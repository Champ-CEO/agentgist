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

    llm = create_llm(Config.Model.DEFAULT)
    selected_posts = filter_posts(llm, query, posts).result()
    logger.debug(f"Selected posts: {[p.model_dump() for p in selected_posts]}")

    for post in selected_posts:
        post.analysis = analyze_post(llm, query, post).result()
        logger.debug(post.analysis.model_dump())

    report_llm = create_llm(Config.Model.REPORT_WRITER)
    report = write_report(report_llm, query, selected_posts).result()
    logger.debug(f"Report: {report.model_dump()}")

    report.references = selected_posts
    return report
