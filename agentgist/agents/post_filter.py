from typing import Any, List

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.func import task

from agentgist.data import Comment, Post
from agentgist.tools import search_documents

SEARCH_PROMPT = """
You are an assistant searching for Reddit posts.
Given a query, please use the search_documents tool to find relevant posts.
Query: {query}

Think step by step about what kinds of posts would be relevant. 
Use the search_documents tool to search for posts matching the query.
"""


POST_TEMPLATE = """
Title: {title}
Author: {author}
Score: {score}
Category: {category}
Content: {text}
Comments:
{comments}
"""


COMMENT_TEMPLATE = """
{text}
    {replies}
"""


def _create_comment_text(comment: Comment) -> str:
    if comment.replies:
        replies_text = "\n".join([_create_comment_text(reply) for reply in comment.replies])
        return f"""
{comment.author}: {comment.text}
Replies:
{replies_text}
"""
    else:
        return f"{comment.author}: {comment.text}"


def _create_post_text(post: Post) -> str:
    return POST_TEMPLATE.format(
        title=post.title,
        author=post.author,
        score=post.score,
        category=post.category,
        text=post.text,
        comments="\n".join([_create_comment_text(comment) for comment in post.comments]),
    )


def _create_documents_from_posts(posts: List[Post]) -> List[Document]:
    return [
        Document(
            page_content=_create_post_text(post),
            metadata={"permalink": post.permalink},
        )
        for post in posts
    ]


def _execute_tool_search(llm: BaseChatModel, query: str) -> dict:
    messages = [HumanMessage(content=SEARCH_PROMPT.format(query=query))]
    llm_response = llm.invoke(messages)

    # Check if there are any tool calls
    if not hasattr(llm_response, "tool_calls") or not llm_response.tool_calls:
        # If no tool calls, create a default tool call for search_documents
        return {"name": "search_documents", "args": {"query": query}}

    return llm_response.tool_calls[0]


def _execute_tool(tool_call: dict, tools: List[BaseTool]) -> Any:
    tools_by_name = {t.name: t for t in tools}
    tool_to_call = tools_by_name[tool_call["name"]]
    return tool_to_call.invoke(tool_call["args"])


@task
def filter_posts(llm: BaseChatModel, query: str, posts: List[Post]) -> List[Post]:
    tools = [search_documents]
    llm_with_tools = llm.bind_tools(tools)

    tool_call = _execute_tool_search(llm_with_tools, query)
    tool_call["args"]["documents"] = _create_documents_from_posts(posts)

    found_documents = _execute_tool(tool_call, tools)
    permalinks = set([doc.metadata["permalink"] for doc in found_documents])
    return [post for post in posts if post.permalink in permalinks]
