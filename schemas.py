"""
schemas.py
----------
Pydantic request and response schema definitions for the SpamGuard API.

These schemas serve three purposes within the FastAPI framework:
  1. Input validation  — incoming JSON payloads are automatically validated
                         against the declared field types and constraints
                         before the endpoint handler is invoked.
  2. Output serialisation — response models ensure that only declared fields
                            are included in HTTP responses, preventing
                            accidental data leakage.
  3. Documentation — FastAPI reads these schemas to auto-generate the
                     OpenAPI specification visible at /docs.
"""

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class PredictRequest(BaseModel):
    """
    Request body schema for the POST /predict endpoint.

    Attributes
    ----------
    email_text : str
        The raw email body text to be submitted for classification.
        Must be a non-empty string.
    email_id : str, optional
        A caller-supplied identifier for the email. Used to correlate
        the classification result with the originating message in
        downstream systems. Defaults to 'unknown' if not provided.
    """

    email_text: str
    email_id:   Optional[str] = 'unknown'


class BatchRequest(BaseModel):
    """
    Request body schema for the POST /predict/batch endpoint.

    Attributes
    ----------
    emails : List[str]
        An ordered list of raw email body strings to be submitted for
        asynchronous batch classification via the Celery task queue.
    """

    emails: List[str]


class PredictResponse(BaseModel):
    """
    Response body schema for the POST /predict endpoint.

    Attributes
    ----------
    email_id : str
        Echo of the caller-supplied email identifier from the request.
    result : str
        Classification verdict. One of: 'spam', 'ham', 'uncertain'.
    confidence : float
        Spam class probability in the range [0.0, 1.0], rounded to
        four decimal places.
    tier_used : int
        The cascade tier that produced the final verdict:
        1 = SpamAssassin, 2 = DistilBERT-FGM, 3 = Flan-T5.
    """

    email_id:   str
    result:     str
    confidence: float
    tier_used:  int


class HistoryItem(BaseModel):
    """
    Response body schema for individual records returned by GET /history.

    Mirrors the PredictionLog ORM model, projecting a subset of fields
    appropriate for external consumption. The from_attributes configuration
    enables direct construction from SQLAlchemy model instances.

    Attributes
    ----------
    id : int
        Auto-incremented primary key of the log record.
    email_id : str
        Caller-supplied identifier of the classified email.
    result : str
        Classification verdict stored at the time of the request.
    confidence : float
        Spam probability recorded at the time of the request.
    timestamp : datetime
        UTC timestamp of the original classification request.
    """

    id:         int
    email_id:   str
    result:     str
    confidence: float
    timestamp:  datetime

    class Config:
        # Permits Pydantic to read field values directly from SQLAlchemy
        # ORM model instances returned by database queries.
        from_attributes = True
