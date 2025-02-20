from langgraph.func import task
from langgraph.types import interrupt


@task
def request_filter_query(posts):
    feedback = interrupt({"posts": posts})
    return feedback["filter_query"]
