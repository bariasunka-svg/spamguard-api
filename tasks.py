"""
tasks.py
--------
Celery asynchronous task definitions for the SpamGuard API.

Defines the Celery application instance and the background task responsible
for processing batch spam classification requests. The task queue decouples
the HTTP request lifecycle from the classification workload, allowing the
FastAPI application to return an immediate response while the Celery worker
processes the submitted emails asynchronously in a separate process.

Architecture
------------
The message broker and result backend are both provided by Redis. When
process_batch.delay() is called from an endpoint handler, Celery serialises
the task arguments to JSON, publishes them to the Redis broker, and returns
an AsyncResult object containing the task identifier. A Celery worker process
subscribed to the same broker consumes the message and executes the task.
Results are stored in Redis and retrieved via GET /task/{task_id}.

This architecture implements the Human-in-the-loop asynchronous workflow
described in the SpamGuard system design: callers receive an immediate task
identifier and may poll for results, enabling human review of borderline
classifications before downstream action is taken.
"""

from celery import Celery
from classifier import classify
import os

# ---------------------------------------------------------------------------
# Broker URL resolution
# ---------------------------------------------------------------------------
# The CELERY_BROKER_URL environment variable is injected at runtime via
# docker-compose.yml. Within the Docker network, the Redis service is
# addressable by its service name ('redis'). The fallback value targets
# a locally running Redis instance for development outside Docker.
BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')

# ---------------------------------------------------------------------------
# Celery application instance
# ---------------------------------------------------------------------------
# The first argument ('tasks') must match the module name passed to the
# Celery worker via the -A flag: celery -A tasks worker
# Both the broker and backend are configured to use the same Redis instance.
celery_app = Celery('tasks', broker=BROKER_URL, backend=BROKER_URL)


@celery_app.task
def process_batch(emails: list) -> dict:
    """
    Asynchronous Celery task for batch spam classification.

    Iterates over a list of email body strings, invokes the classify()
    function for each, and aggregates the results into a structured
    summary dictionary. Individual classification failures are caught
    and recorded without interrupting the processing of remaining items.

    This task is dispatched asynchronously via process_batch.delay() from
    the POST /predict/batch endpoint. The caller receives a task identifier
    immediately and retrieves results by polling GET /task/{task_id}.

    Parameters
    ----------
    emails : list of str
        Ordered list of raw email body strings to be classified.

    Returns
    -------
    dict
        A summary dictionary containing:
        - total     (int)  : Total number of emails submitted.
        - processed (int)  : Number of emails for which a result was recorded
                             (includes both successes and handled exceptions).
        - results   (list) : Per-email classification records, each containing:
                             - index      (int)   : Zero-based position in the input list.
                             - email_text (str)   : First 100 characters of the email body.
                             - result     (str)   : 'spam', 'ham', or 'uncertain'.
                             - confidence (float) : Spam probability in [0.0, 1.0].
                             On failure: {'index': int, 'error': str}
    """
    results = []

    for i, email_text in enumerate(emails):
        try:
            classification = classify(email_text)
            results.append({
                'index'     : i,
                # Truncate the stored text to 100 characters to limit
                # result payload size in the Redis backend.
                'email_text': email_text[:100],
                'result'    : classification['result'],
                'confidence': classification['confidence']
            })
        except Exception as exc:
            # Record the exception message without propagating, allowing
            # the remaining emails in the batch to be processed.
            results.append({'index': i, 'error': str(exc)})

    return {
        'total'    : len(emails),
        'processed': len(results),
        'results'  : results
    }
