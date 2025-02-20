from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.func import task

from agentgist.data import Post, Report

SUMMARIZED_POST_TEMPLATE = """
<post>
    <title>{title}</title>
    <text>{text}</text>
    <score>{score}</score>
    <summary>{summary}</summary>
    <topics>{topics}</topics>
    <key_points>{key_points}</key_points>
    <controversies>{controversies}</controversies>
</post>
""".strip()


REPORT_PROMPT = PromptTemplate.from_template(
    """
Write an executive summary on the following:

<posts>
{posts}
</posts>

The report must contain:

- Title that will make the reader want to read the report
- Summary (1-3 paragraphs) - independent overview of the posts and their analysis. Focus on details related to the user query.
- Takeaways (1-2 sentences) - a bullet list with the top 3 takeaways from the analysis (actionable, practical and easy to understand)

The report must respond to the user query:

<user_query>
{user_query}
</user_query>

Ignore posts that are not relevant to the user query.

The report must be written in a way that is immediately presentable to a technical user.
It must be concise and immediately actionable.

Your report must be formatted as a JSON.
""".strip()
)


def _create_post_text(post: Post) -> str:
    analysis = post.analysis
    return SUMMARIZED_POST_TEMPLATE.format(
        title=post.title,
        text=post.text,
        score=post.score,
        summary=analysis.summary,
        topics=analysis.topics,
        sentiment=analysis.sentiment,
        key_points=analysis.key_points,
        controversies=analysis.controversies,
    )


@task
def write_report(llm: BaseChatModel, user_query: str, posts: List[Post]) -> Report:
    llm = llm.with_structured_output(Report, method="json_schema")

    post_texts = [_create_post_text(post) for post in posts]

    return llm.invoke(
        [
            HumanMessage(
                content=REPORT_PROMPT.format(posts="\n".join(post_texts), user_query=user_query),
            ),
        ]
    )
