"""
models.py
---------
SQLAlchemy ORM model definitions for the SpamGuard API.

Defines the database schema for persistent storage of spam classification
results. Each record in the prediction_logs table corresponds to a single
inference request processed by the classification pipeline, providing a
complete audit trail of all verdicts issued by the system.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime
from database import Base
from datetime import datetime


class PredictionLog(Base):
    """
    ORM model representing a single spam classification event.

    Each row records the input identifier, the classification verdict,
    the model's confidence score, the cascade tier that produced the
    final decision, and the UTC timestamp of the request. This table
    serves as the primary audit log for the SpamGuard inference pipeline
    and is queried by the /history and /stats endpoints.

    Table name: prediction_logs
    """

    __tablename__ = 'prediction_logs'

    # Primary key — auto-incrementing integer identifier for each record.
    id = Column(Integer, primary_key=True, index=True)

    # Caller-supplied identifier for the email being classified.
    # Indexed to support efficient lookup by email identifier.
    email_id = Column(String, index=True)

    # Raw email body text submitted for classification. Stored in full
    # to enable retrospective review and audit of classification decisions.
    email_text = Column(String)

    # Classification verdict produced by the inference pipeline.
    # Valid values: 'spam', 'ham', 'uncertain'
    # Indexed to support efficient aggregation queries (e.g. /stats).
    result = Column(String, index=True)

    # Spam class probability returned by the DistilBERT-FGM model,
    # expressed as a floating-point value in the range [0.0, 1.0].
    confidence = Column(Float)

    # Cascade tier that issued the final verdict.
    # 1 = SpamAssassin (rule-based), 2 = DistilBERT-FGM, 3 = Flan-T5
    tier_used = Column(Integer)

    # UTC timestamp recorded at the moment the classification request
    # was received. Defaults to the current UTC time if not supplied.
    timestamp = Column(DateTime, default=datetime.utcnow)
