from pydantic import BaseModel


class RunSummary(BaseModel):
    """End-to-end artifacts captured for a single run."""

    template: str
    context: str
    angles: list[str]
    drafts: list[str]
    selection: str
