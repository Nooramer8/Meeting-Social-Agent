from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class ActionItem(BaseModel):
    model_config = ConfigDict(extra='forbid')

    owner: str = Field(description='Person or team responsible. Use Unknown / غير محدد when not clear.')
    task: str
    deadline: str = Field(description='Specific date if mentioned; otherwise Not specified / غير محدد.')


class MeetingSummary(BaseModel):
    model_config = ConfigDict(extra='forbid')

    language: str = Field(description='Primary output language, for example Arabic or English.')
    title: str
    short_summary: str
    key_points: list[str]
    decisions: list[str]
    action_items: list[ActionItem]
    risks_or_sensitive_items: list[str]
    suggested_facebook_post: str
    suggested_instagram_caption: str
    suggested_hashtags: list[str]


class DraftUpdate(BaseModel):
    facebook_post: str | None = None
    instagram_caption: str | None = None


class PublishResult(BaseModel):
    platform: Literal['facebook', 'instagram']
    remote_id: str | None = None
    raw_response: dict | None = None
