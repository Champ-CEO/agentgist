import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command
from loguru import logger

from agentgist.config import Config, configure_logging
from agentgist.workflow import ReportWorkflowInput, write_report_workflow

load_dotenv()

configure_logging()
logger.add(Config.LOG_FILE)


WORKFLOW_CONFIG = {"configurable": {"thread_id": 42}}

INTERRUPT_KEY = "__interrupt__"

POST_LIST_TEMPLATE = """
Those are the posts I found at *{subreddit}*:

{posts}

**What do you want to focus on?**
""".strip()

POST_ITEM_TEMPLATE = """
- ({score}) **{title}** ({n_comments} comments)
""".strip()

REPORT_TEMPLATE = """
#### {title}

Takeaways:

{takeaways}

Summary:

{summary}

References:

{references}
""".strip()


st.set_page_config(
    page_title="AgentGist",
    page_icon="😎",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("AgentGist")
st.subheader("Decoding Reddit, one post at a time.")
st.markdown("Analyze Reddit posts from a subreddit of your choice.")


if "is_subreddit_selected" not in st.session_state:
    st.session_state["is_subreddit_selected"] = False

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    avatar = "🐧" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def _format_report(report):
    return REPORT_TEMPLATE.format(
        title=report.title,
        takeaways="\n\n".join([f"- {takeaway}" for takeaway in report.takeaways]),
        summary=report.summary,
        references="\n\n".join(
            [
                f"- [{post.title}](https://www.reddit.com{post.permalink})"
                for post in report.references
            ]
        ),
    )


def _format_post_list(subreddit, posts):
    post_texts = "\n\n".join(
        [
            POST_ITEM_TEMPLATE.format(
                score=post.score, title=post.title, n_comments=post.n_comments
            )
            for post in posts
        ]
    )
    return POST_LIST_TEMPLATE.format(subreddit=subreddit, posts=post_texts)


def on_chat_input():
    user_query = st.session_state.user_input
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Writing report...", show_time=True):
        command = Command(resume={"filter_query": user_query})
        report = write_report_workflow.invoke(command, WORKFLOW_CONFIG)
        st.session_state.messages.append({"role": "assistant", "content": _format_report(report)})


def on_choose_subreddit():
    with st.spinner("Fetching Reddit data...", show_time=True):
        subreddit = st.session_state.subreddit.strip("/")
        workflow_input = ReportWorkflowInput(subreddit, st.session_state.max_posts)
        updates = write_report_workflow.invoke(
            workflow_input, WORKFLOW_CONFIG, stream_mode="updates"
        )
        interrupt_update = [update for update in updates if INTERRUPT_KEY in update][0]

        data = interrupt_update[INTERRUPT_KEY][0]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": _format_post_list(subreddit, data.value["posts"]),
            }
        )

        st.chat_input(
            "Focus on the top 3 posts about...", on_submit=on_chat_input, key="user_input"
        )


def choose_subreddit_form():
    holder = st.empty()
    with holder.container():
        with st.form(key="reddit_form"):
            st.write("Choose your adventure")
            st.text_input(
                "Subreddit",
                value="r/LocalLLaMA",
                placeholder="e.g. r/LocalLLaMA",
                key="subreddit",
            )
            st.number_input(
                "Maximum posts",
                value=5,
                min_value=1,
                step=1,
                max_value=20,
                key="max_posts",
            )
            submit_button = st.form_submit_button("Download posts", on_click=on_choose_subreddit)

        if submit_button:
            holder.empty()
            st.session_state["is_subreddit_selected"] = True


if not st.session_state["is_subreddit_selected"]:
    choose_subreddit_form()
