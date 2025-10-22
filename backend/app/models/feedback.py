from sqlmodel import SQLModel, Field
from datetime import datetime

class Feedback(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)