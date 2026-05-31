# tasks.py
from celery import Celery
from classifier import classify
import os

BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')

celery_app = Celery('tasks', broker=BROKER_URL, backend=BROKER_URL)

@celery_app.task
def process_batch(emails: list) -> dict:
    results = []
    for i, email_text in enumerate(emails):
        try:
            result = classify(email_text)
            results.append({
                'index': i,
                'email_text': email_text[:100],
                'result': result['result'],
                'confidence': result['confidence']
            })
        except Exception as e:
            results.append({'index': i, 'error': str(e)})
    return {
        'total': len(emails),
        'processed': len(results),
        'results': results
    }