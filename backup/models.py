# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
from datetime import datetime

class PredictionLog(Base):
    __tablename__ = 'prediction_logs'

    id         = Column(Integer, primary_key=True, index=True)
    email_id   = Column(String, index=True)
    email_text = Column(String)
    result     = Column(String)   # 'spam', 'ham', 'uncertain'
    confidence = Column(Float)    # 0.0 to 1.0
    tier_used  = Column(Integer)  # 1, 2, or 3
    timestamp  = Column(DateTime, default=datetime.utcnow)
