from datetime import datetime
from typing import ForwardRef, List, Literal, Optional

from pydantic import BaseModel, Field


class PostAnalysis(BaseModel):
    summary: str = Field(..., description="Summary of post and comments into a few sentences")
    key_points: List[str] = Field([], description="Key points from the post and comments")
    topics: List[str] = Field([], description="Main topics discussed in the comments")
    controversies: List[str] = Field([], description="Controversial takeaways from the comments")
    sentiment: Literal[
        "happiness", "anger", "sadness", "fear", "surprise", "disgust", "trust", "anticipation"
    ] = Field(..., description="Overall emotional sentiment of the post and comments")


Comment = ForwardRef("Comment")


class Comment(BaseModel):
    text: str
    author: str
    score: int
    replies: List[Comment] = []


Comment.model_rebuild()


class Post(BaseModel):
    permalink: str
    title: str
    text: str
    author: str
    category: Optional[str]
    score: int
    upvote_ratio: float
    n_comments: int
    url_domain: str
    created_at: datetime
    comments: List[Comment] = []
    analysis: Optional[PostAnalysis] = None


class Report(BaseModel):
    title: str = Field(..., description="Title of the report")
    takeaways: List[str] = Field([], description="Top 3 takeaways from the analysis")
    summary: str = Field(
        ...,
        description="Summary (1-3 paragraphs) - independent overview of the posts and their analysis. Focus on details related to the user query.",
    )
    references: Optional[List[Post]] = Field(
        [], description="List of posts referenced in the report"
    )
