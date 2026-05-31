# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime

from schemas import PredictRequest, PredictResponse, BatchRequest, HistoryItem
from database import get_db, engine
from models import Base, PredictionLog
from classifier import classify
from tasks import process_batch

# Create DB tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title='SpamGuard API',
    description='Three-tier spam detection — DistilBERT-FGM thesis model',
    version='1.0.0'
)

@app.get('/health')
async def health():
    return {'status': 'ok', 'model': 'DistilBERT-FGM-seed456'}

@app.post('/predict', response_model=PredictResponse)
async def predict(req: PredictRequest, db: Session = Depends(get_db)):
    result = classify(req.email_text)
    log = PredictionLog(
        email_id=req.email_id, result=result['result'],
        confidence=result['confidence'], timestamp=datetime.utcnow()
    )
    db.add(log); db.commit()
    return PredictResponse(email_id=req.email_id, **result)

@app.get('/history', response_model=list[HistoryItem])
async def history(limit: int = 20, db: Session = Depends(get_db)):
    return db.query(PredictionLog).order_by(
        PredictionLog.timestamp.desc()).limit(limit).all()

@app.post('/predict/batch')
async def predict_batch(req: BatchRequest):
    task = process_batch.delay(req.emails)
    return {'task_id': task.id, 'status': 'processing',
            'email_count': len(req.emails)}

@app.get('/task/{task_id}')
async def get_task_status(task_id: str):
    from tasks import celery_app
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)
    return {'task_id': task_id, 'status': result.state,
            'result': result.result if result.ready() else None}

@app.get('/stats')
async def stats(db: Session = Depends(get_db)):
    total = db.query(PredictionLog).count()
    spam  = db.query(PredictionLog).filter(PredictionLog.result=='spam').count()
    return {'total': total, 'spam': spam, 'ham': total - spam}
