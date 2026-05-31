"""
main.py
-------
FastAPI application entry point for the SpamGuard spam detection API.

Defines the FastAPI application instance and all HTTP endpoint handlers.
At startup, the application initialises the SQLAlchemy database schema and
loads the DistilBERT-FGM classification model into memory via the classifier
module. All endpoints are asynchronous to support concurrent request handling
under the Uvicorn ASGI server.

Endpoints
---------
GET  /health          — Service health check and model identifier
POST /predict         — Single-email spam classification with DB logging
GET  /history         — Paginated retrieval of past classification records
POST /predict/batch   — Asynchronous batch classification via Celery
GET  /task/{task_id}  — Status and result polling for batch tasks
GET  /stats           — Aggregate classification statistics

The three-tier cascade architecture is partially implemented here:
Tier 2 (DistilBERT-FGM) is active for all /predict requests. Tier 1
(SpamAssassin) and Tier 3 (Flan-T5) integration points are reserved
for future extension.
"""

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from schemas import PredictRequest, PredictResponse, BatchRequest, HistoryItem
from database import get_db, engine
from models import Base, PredictionLog
from classifier import classify
from tasks import process_batch

# ---------------------------------------------------------------------------
# Database schema initialisation
# ---------------------------------------------------------------------------
# Creates all tables defined in models.py if they do not already exist.
# This is called once at module import time, before any requests are served.
Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title='SpamGuard API',
    description=(
        'Three-tier cascade spam detection service backed by a DistilBERT-FGM '
        'model trained on four benchmark email corpora (Enron, LingSpam, '
        'SpamAssassin, TREC-07) as part of a Master\'s thesis at CYUT, Taiwan.'
    ),
    version='1.0.0'
)


# ---------------------------------------------------------------------------
# Endpoint: GET /health
# ---------------------------------------------------------------------------
@app.get('/health', summary='Service health check')
async def health():
    """
    Returns the operational status of the API and the active model identifier.

    This endpoint is intended for use by container orchestration systems,
    load balancers, and monitoring tools to verify that the service is
    running and that the classification model has been loaded successfully.
    """
    return {'status': 'ok', 'model': 'DistilBERT-FGM-seed456'}


# ---------------------------------------------------------------------------
# Endpoint: POST /predict
# ---------------------------------------------------------------------------
@app.post('/predict', response_model=PredictResponse, summary='Classify a single email')
async def predict(req: PredictRequest, db: Session = Depends(get_db)):
    """
    Classifies a single email and persists the result to the database.

    Invokes the Tier 2 DistilBERT-FGM classifier, writes the result to the
    prediction_logs table via SQLAlchemy, and returns the structured verdict
    to the caller. The database session is supplied via FastAPI dependency
    injection and is closed automatically after the response is sent.
    """
    # Invoke the Tier 2 classifier to obtain a verdict and confidence score.
    result = classify(req.email_text)

    # Persist the classification result as an audit record in the database.
    log = PredictionLog(
        email_id   = req.email_id,
        email_text = req.email_text,
        result     = result['result'],
        confidence = result['confidence'],
        tier_used  = result['tier_used'],
        timestamp  = datetime.utcnow()
    )
    db.add(log)
    db.commit()

    return PredictResponse(email_id=req.email_id, **result)


# ---------------------------------------------------------------------------
# Endpoint: GET /history
# ---------------------------------------------------------------------------
@app.get('/history', response_model=list[HistoryItem], summary='Retrieve classification history')
async def history(limit: int = 20, db: Session = Depends(get_db)):
    """
    Returns the most recent classification records in reverse chronological order.

    Parameters
    ----------
    limit : int, optional
        Maximum number of records to return. Defaults to 20.
    """
    return (
        db.query(PredictionLog)
          .order_by(PredictionLog.timestamp.desc())
          .limit(limit)
          .all()
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /predict/batch
# ---------------------------------------------------------------------------
@app.post('/predict/batch', summary='Submit a batch of emails for asynchronous classification')
async def predict_batch(req: BatchRequest):
    """
    Dispatches a batch classification job to the Celery task queue.

    Returns a task identifier immediately without waiting for classification
    to complete. The caller may poll GET /task/{task_id} to retrieve results
    once processing is finished. This non-blocking pattern is central to the
    Human-in-the-loop workflow, allowing human review before action is taken.
    """
    task = process_batch.delay(req.emails)
    return {
        'task_id'    : task.id,
        'status'     : 'processing',
        'email_count': len(req.emails)
    }


# ---------------------------------------------------------------------------
# Endpoint: GET /task/{task_id}
# ---------------------------------------------------------------------------
@app.get('/task/{task_id}', summary='Poll the status of an asynchronous batch task')
async def get_task_status(task_id: str):
    """
    Retrieves the current state and result of a Celery background task.

    Parameters
    ----------
    task_id : str
        The unique task identifier returned by POST /predict/batch.

    Returns a status of 'PENDING', 'STARTED', 'SUCCESS', or 'FAILURE'.
    The result field is populated only when status is 'SUCCESS'.
    """
    from tasks import celery_app
    from celery.result import AsyncResult

    async_result = AsyncResult(task_id, app=celery_app)
    return {
        'task_id': task_id,
        'status' : async_result.state,
        'result' : async_result.result if async_result.ready() else None
    }


# ---------------------------------------------------------------------------
# Endpoint: GET /stats
# ---------------------------------------------------------------------------
@app.get('/stats', summary='Retrieve aggregate classification statistics')
async def stats(db: Session = Depends(get_db)):
    """
    Returns aggregate counts of all classifications recorded in the database.

    Queries the prediction_logs table to compute the total number of
    processed emails and the breakdown between spam and ham verdicts.
    """
    total = db.query(PredictionLog).count()
    spam  = db.query(PredictionLog).filter(PredictionLog.result == 'spam').count()
    return {
        'total': total,
        'spam' : spam,
        'ham'  : total - spam
    }
