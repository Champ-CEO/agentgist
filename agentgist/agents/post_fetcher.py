from datetime import datetime, timezone
from typing import List

import requests
from langgraph.func import task

from agentgist.data import Comment, Post

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RedditScraper/1.0; +https://superawesomeness.com/bot)"
}


def _fetch_posts(subreddit, limit: int):
    url = f"https://www.reddit.com/{subreddit}/.json?limit={limit}"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(f"Error fetching subreddit {subreddit}: {response.status_code}")
    return response.json()


def _fetch_comments(post_permalink, limit: int):
    url = f"https://www.reddit.com/{post_permalink}.json?limit={limit}"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise Exception(f"Error fetching comments from {post_permalink}: {response.status_code}")
    return response.json()


def _extract_posts(subreddit_data) -> List[Post]:
    posts = []

    for idx, post in enumerate(subreddit_data["data"]["children"]):
        post_data = post["data"]
        post = Post(
            id=str(idx + 1),
            title=post_data["title"],
            text=post_data["selftext"],
            author=post_data["author"],
            category=post_data["link_flair_text"],
            score=post_data["score"],
            upvote_ratio=post_data["upvote_ratio"],
            permalink=post_data["permalink"],
            url_domain="" if post_data["domain"] == "self.LocalLLaMA" else post_data["domain"],
            n_comments=post_data["num_comments"],
            created_at=datetime.fromtimestamp(post_data["created_utc"], tz=timezone.utc),
        )
        posts.append(post)
    return posts


def _extract_comments(
    data, max_depth: int = -1, current_depth: int = 0, min_score: int = 2
) -> List[Comment]:
    comments = []

    if isinstance(data, list):
        for item in data:
            comments.extend(_extract_comments(item, max_depth, current_depth))
        return comments

    if not isinstance(data, dict):
        return comments

    if "data" in data and "children" in data["data"]:
        for child in data["data"]["children"]:
            kind = child.get("kind")
            child_data = child.get("data", {})

            if kind != "t1":
                continue

            score = child_data["score"]
            if score < min_score:
                continue

            replies = child_data.get("replies")
            comment_replies = []
            if replies and isinstance(replies, dict):
                if max_depth == -1 or current_depth < max_depth:
                    comment_replies = _extract_comments(replies, max_depth, current_depth + 1)

            comments.append(
                Comment(
                    text=child_data["body"],
                    author=child_data["author"],
                    score=score,
                    replies=comment_replies,
                )
            )
    return comments


@task
def fetch_subreddit_posts(subreddit, max_posts=10, max_comments_per_post=30) -> List[Post]:
    posts_data = _fetch_posts(subreddit, limit=max_posts)
    posts = _extract_posts(posts_data)
    for post in posts:
        comments_data = _fetch_comments(post.permalink, max_comments_per_post)
        comments = _extract_comments(comments_data)
        post.comments = comments
    return posts
