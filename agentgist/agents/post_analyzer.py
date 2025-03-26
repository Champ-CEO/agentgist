from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langgraph.func import task
import json

from agentgist.data import Comment, Post, PostAnalysis

ANALYZE_POST_PROMPT = PromptTemplate.from_template(
    """
Analyze the post and associated comments:

<post>
{post}
</post>

<comments>
{comments}
</comments>

The analysis must contain:

- Summary of post and comments into a few sentences (e.g. A growing number of people are...)
- Key points from the post and comments (top 3)
- Main topics discussed in the post and comments (top 3)
- Controversial takeaways from the comments (top 3)
- Sentiment of the post and comments (choose one from happiness, anger, sadness, fear, surprise, disgust, trust, anticipation)

Your analysis should respond to the user query:

<user_query>
{user_query}
</user_query>
""".strip()
)

POST_TEMPLATE = """
<post>
    <title>{title}</title>
    <text>{text}</text>
    <category>{category}</category>
    <author>{author}</author>
    <n_comments>{n_comments}</n_comments>
    <score>{score}</score>
    <upvote_ratio>{upvote_ratio}</upvote_ratio>
    <url_domain>{url_domain}</url_domain>
    <created_at>{created_at}</created_at>
</post>
""".strip()

COMMENT_TEMPLATE = """
<comment>
    <text>{text}</text>
    <author>{author}</author>
    <score>{score}</score>
    <replies>{replies}</replies>
</comment>
""".strip()


def _create_post_text(post: Post) -> str:
    return POST_TEMPLATE.format(
        title=post.title,
        text=post.text,
        category=post.category,
        author=post.author,
        n_comments=post.n_comments,
        score=post.score,
        upvote_ratio=post.upvote_ratio,
        url_domain=post.url_domain,
        created_at=post.created_at.isoformat(),
    )


def _create_comment_text(comment: Comment) -> str:
    return COMMENT_TEMPLATE.format(
        text=comment.text,
        author=comment.author,
        score=comment.score,
        replies="\n".join([_create_comment_text(reply) for reply in comment.replies]),
    )


@task
def analyze_post(llm: BaseChatModel, user_query: str, post: Post) -> PostAnalysis:
    # For Groq compatibility, we will use a system message to request JSON output
    # instead of using with_structured_output with method="json_mode"

    schema = PostAnalysis.schema()
    schema_str = json.dumps(schema, indent=2)

    system_message = SystemMessage(
        content=f"You must respond with a JSON object that conforms to this schema: {schema_str}"
    )

    human_message = HumanMessage(
        content=ANALYZE_POST_PROMPT.format(
            post=_create_post_text(post),
            comments="\n".join([_create_comment_text(comment) for comment in post.comments]),
            user_query=user_query,
        ),
    )

    # Get the raw JSON response
    response = llm.invoke([system_message, human_message])
    response_content = response.content

    # Parse the JSON response into a PostAnalysis object
    try:
        # Handle potential JSON in markdown code blocks
        if "```json" in response_content:
            json_str = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_str = response_content.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_content

        # Parse the JSON and create a PostAnalysis object
        json_data = json.loads(json_str)
        return PostAnalysis(**json_data)
    except Exception as e:
        # Fallback with minimal valid data if JSON parsing fails
        return PostAnalysis(
            summary=f"Error parsing response: {str(e)}. Original response: {response_content[:100]}...",
            key_points=["Analysis failed", "JSON parsing error", "Check logs"],
            topics=["error"],
            controversies=["none"],
            sentiment="neutral",
        )
