# schemas.py
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Request: what the caller sends to POST /predict
class PredictRequest(BaseModel):
    email_text: str
    email_id:   Optional[str] = 'unknown'

# Request: what the caller sends to POST /predict/batch
class BatchRequest(BaseModel):
    emails: List[str]

# Response: what POST /predict returns
class PredictResponse(BaseModel):
    email_id:   str
    result:     str   # 'spam', 'ham', or 'uncertain'
    confidence: float # 0.0 to 1.0
    tier_used:  int   # 1, 2, or 3

# Response: what GET /history returns (one row)
class HistoryItem(BaseModel):
    id:         int
    email_id:   str
    result:     str
    confidence: float
    timestamp:  datetime
    class Config:
        from_attributes = True  # allows reading from SQLAlchemy objects
