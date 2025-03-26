from typing import List
import json

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
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
        key_points=analysis.key_points,
        controversies=analysis.controversies,
    )


@task
def write_report(llm: BaseChatModel, user_query: str, posts: List[Post]) -> Report:
    # For Groq compatibility, we will use a system message to request JSON output
    # instead of using with_structured_output with method="json_mode"

    schema = Report.schema()
    schema_str = json.dumps(schema, indent=2)

    system_message = SystemMessage(
        content=f"You must respond with a JSON object that conforms to this schema: {schema_str}"
    )

    post_texts = [_create_post_text(post) for post in posts]

    human_message = HumanMessage(
        content=REPORT_PROMPT.format(posts="\n".join(post_texts), user_query=user_query),
    )

    # Get the raw JSON response
    response = llm.invoke([system_message, human_message])
    response_content = response.content

    # Parse the JSON response into a Report object
    try:
        # Handle potential JSON in markdown code blocks
        if "```json" in response_content:
            json_str = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_str = response_content.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_content

        # Parse the JSON and create a Report object
        json_data = json.loads(json_str)
        return Report(**json_data)
    except Exception as e:
        # Fallback with minimal valid data if JSON parsing fails
        return Report(
            title=f"Error Generating Report: {str(e)[:50]}...",
            summary=f"There was an error parsing the model response. Original content: {response_content[:100]}...",
            takeaways=[
                "Report generation failed",
                "Try a different query",
                "Check logs for details",
            ],
        )
