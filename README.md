# SpamGuard API

> A production-grade spam detection REST API built on a DistilBERT-FGM model we trained across four benchmark email corpora as part of our Master's thesis in Information Management at Chaoyang University of Technology (CYUT), Taiwan.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=flat-square&logo=docker&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?style=flat-square)
![Celery](https://img.shields.io/badge/Celery-5.4-37814A?style=flat-square&logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7.0-DC382D?style=flat-square&logo=redis&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [The Classification Model](#the-classification-model)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Academic Context](#academic-context)
- [Roadmap](#roadmap)

---

## Overview

SpamGuard API is the deployment layer for a spam classification model developed during our Master's research. It exposes a fine-tuned DistilBERT model — trained with Fast Gradient Method (FGM) adversarial regularisation across four real-world email corpora — as a fully containerised REST service.

The service accepts raw email text and returns a structured verdict (`spam`, `ham`, or `uncertain`) alongside a calibrated confidence score. Emails classified as `uncertain` are architecturally designed to escalate to a Flan-T5 large language model for human review — implementing the Human-in-the-loop workflow designed in our thesis cascade architecture.

**What this project demonstrates:**

- Deploying a research ML model as a production-ready API using FastAPI
- Persistent classification audit logging via SQLAlchemy ORM
- Asynchronous batch processing via Celery and Redis
- Full OpenAPI documentation auto-generated at `/docs`
- A complete three-container stack deployable with a single command

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT                               │
│              (Browser / Postman / Mail Server)              │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONTAINER: app                            │
│                FastAPI + Uvicorn (:8000)                    │
│                                                             │
│  POST /predict      ──► DistilBERT-FGM Inference           │
│  POST /predict/batch──► Celery Task Dispatch               │
│  GET  /history      ──► SQLAlchemy Query                   │
│  GET  /task/{id}    ──► Celery Result Retrieval            │
│  GET  /stats        ──► Aggregate DB Query                 │
│  GET  /health       ──► Service Status                     │
└──────────┬──────────────────────┬───────────────────────────┘
           │ SQLAlchemy ORM       │ Celery .delay()
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────────────────┐
│  spam.db         │   │  CONTAINER: redis                    │
│  (SQLite)        │   │  Redis 7 — Message Broker (:6379)    │
│                  │   └──────────────┬───────────────────────┘
│  prediction_logs │                  │ Task consumption
└──────────────────┘                  ▼
                       ┌──────────────────────────────────────┐
                       │  CONTAINER: worker                   │
                       │  Celery Worker                       │
                       │  tasks.process_batch()               │
                       │  DistilBERT-FGM Inference (batch)   │
                       └──────────────────────────────────────┘
```

### Three-Tier Cascade Design

In the thesis, the study designed a three-tier cascade architecture to balance speed, accuracy, and operational cost. This API implements Tier 2.

| Tier | Component | Trigger Condition | Role |
|------|-----------|-------------------|------|
| 1 | SpamAssassin (rule-based) | Score > 8.0 or < 2.0 | Fast triage for obvious cases |
| **2** | **DistilBERT-FGM** *(this API)* | **All uncertain emails from Tier 1** | **Primary AI classifier** |
| 3 | Flan-T5-base (LLM) | Confidence ∈ [0.3, 0.7] | Human-in-the-loop escalation |

Tier 1 and Tier 3 integration are reserved for future releases. This repository implements Tier 2 and the async escalation workflow via Celery.

---

## The Classification Model

### Model Identity

| Parameter | Value |
|-----------|-------|
| Base architecture | `distilbert-base-uncased` |
| Training method | Fast Gradient Method (FGM) adversarial training |
| Maximum token length | 256 |
| Training corpora | Enron, LingSpam, SpamAssassin, TREC-07 (multi-corpus) |
| Random seed | 456 |
| Checkpoint | `p3_full_fgm_ml256_s456/best/` |
| Parameters | ~66 million |

### Why We Chose This Model

We systematically trained and evaluated 30 model configurations — 2 training methods × 3 token lengths × 5 random seeds — under Leave-One-Corpus-Out (LOCO) cross-domain testing. We chose **False Positive Rate (FPR)** as our primary optimisation target because blocking a legitimate email is a more costly error than missing spam in a production security system.

The `p3_full_fgm_ml256_s456` configuration achieved the lowest cross-domain FPR across all evaluation conditions, demonstrating robust generalisation to unseen email distributions. I selected this checkpoint for deployment.

**Why FGM adversarial training?** During training, FGM exposes the model to adversarially perturbed text examples, improving its resistance to subtle variations — a property we designed specifically for cybersecurity deployment contexts where adversaries actively attempt to evade detection.

**Why 256 tokens?** 128 tokens truncated too many emails before they revealed their spam signals. 512 tokens doubled memory usage with no measurable accuracy gain. 256 is the sweet spot we identified experimentally.

### Inference Decision Thresholds

```
p_spam > 0.70  →  verdict: "spam"       (high confidence — block)
p_spam < 0.30  →  verdict: "ham"        (high confidence — deliver)
0.30 ≤ p_spam ≤ 0.70  →  verdict: "uncertain"  (escalate to Tier 3)
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| API framework | FastAPI 0.111 | Async HTTP endpoints, automatic OpenAPI documentation |
| ASGI server | Uvicorn 0.29 | High-performance async server |
| ML inference | Transformers 4.41 + PyTorch 2.3 | DistilBERT-FGM model loading and forward pass |
| Database ORM | SQLAlchemy 2.0 | Classification audit log, prediction history |
| Task queue | Celery 5.4 | Asynchronous batch processing |
| Message broker | Redis 7 | Celery broker and result backend |
| Data validation | Pydantic 2.7 | Request/response schema validation |
| Containerisation | Docker + docker-compose | Reproducible multi-container deployment |

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- The trained model checkpoint placed at local repository `./models/p3_full_fgm_ml256_s456/best/`.
  *(Required files: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`)*

> The model is not included in this repository due to its size (~260 MB).
> It is available on request at **bariasunka@gmail.com**.

### 1. Clone the repository

```bash
git clone https://github.com/bariasunka-svg/spamguard-api.git
cd spamguard-api
```

### 2. Place the model

```
spamguard-api/
└── models/
    └── p3_full_fgm_ml256_s456/
        └── best/
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            └── tokenizer_config.json
```

### 3. Start the full stack

```bash
docker compose up
```

This starts three containers: the FastAPI application server, the Celery worker, and the Redis message broker. The model loads into memory during startup (~30 seconds on first run).

### 4. Access the API

| Interface | URL |
|-----------|-----|
| Interactive API documentation (Swagger UI) | http://localhost:8000/docs |
| OpenAPI schema | http://localhost:8000/openapi.json |
| Health check | http://localhost:8000/health |

### 5. Run a quick test

```bash
# Spam example
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Congratulations! You have won $1,000,000. Click here now!", "email_id": "test001"}'

# Expected response
{
  "email_id":   "test001",
  "result":     "spam",
  "confidence": 0.9981,
  "tier_used":  2
}
```

```bash
# Ham example
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Hi, the project meeting is confirmed for Thursday at 3pm.", "email_id": "test002"}'

# Expected response
{
  "email_id":   "test002",
  "result":     "ham",
  "confidence": 0.0007,
  "tier_used":  2
}
```

---

## API Reference

### `GET /health`
Returns the operational status of the service and the active model identifier.

**Response**
```json
{"status": "ok", "model": "DistilBERT-FGM-seed456"}
```

---

### `POST /predict`
Classifies a single email and persists the result to the audit database.

**Request body**
```json
{
  "email_text": "string (required)",
  "email_id":   "string (optional, default: 'unknown')"
}
```

**Response**
```json
{
  "email_id":   "string",
  "result":     "spam | ham | uncertain",
  "confidence": 0.0,
  "tier_used":  2
}
```

---

### `GET /history`
Returns recent classification records in reverse chronological order.

**Query parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 20 | Maximum number of records to return |

---

### `POST /predict/batch`
Dispatches a batch of emails for asynchronous classification via the Celery task queue. Returns immediately with a task identifier — no waiting.

**Request body**
```json
{"emails": ["email body 1", "email body 2", "..."]}
```

**Response**
```json
{
  "task_id":     "string",
  "status":      "processing",
  "email_count": 0
}
```

---

### `GET /task/{task_id}`
Polls the status and result of an asynchronous batch classification task.

**Response (complete)**
```json
{
  "task_id": "string",
  "status":  "SUCCESS",
  "result": {
    "total":     2,
    "processed": 2,
    "results": [
      {"index": 0, "email_text": "...", "result": "spam", "confidence": 0.9981},
      {"index": 1, "email_text": "...", "result": "ham",  "confidence": 0.0007}
    ]
  }
}
```

---

### `GET /stats`
Returns aggregate classification counts across all records in the audit database.

**Response**
```json
{"total": 42, "spam": 31, "ham": 11}
```

---

## Project Structure

```
spamguard-api/
├── classifier.py        # Model loading and inference (DistilBERT-FGM)
├── database.py          # SQLAlchemy engine, session factory, get_db() dependency
├── models.py            # ORM model — PredictionLog table definition
├── schemas.py           # Pydantic request/response schema definitions
├── tasks.py             # Celery application instance and process_batch() task
├── main.py              # FastAPI application and endpoint handlers
├── requirements.txt     # Pinned Python dependencies
├── Dockerfile           # Container image build instructions
├── docker-compose.yml   # Multi-container orchestration (app, worker, redis)
└── .gitignore           # Excludes venv, model weights, database, and cache files
```

---

## Academic Context

We built this project as part of our Master's research at Chaoyang University of Technology (CYUT), Taiwan, where we investigated cross-domain email spam detection using adversarially trained transformer models.

**Thesis title:** *Disentangling Data Diversity and Adversarial Robustness in Cross-Domain Email Spam Detection: Empirical Evidence from Multi-Corpus DistilBERT Training*  
**Author:** Bari Asunka  
**Advisor:** 陳金鈴 博士 (Chin-Ling Chen)
**Institution:** Chaoyang University of Technology (CYUT), Taichung, Taiwan  
**Degree:** Master of Science in Information Management

### How Our Thesis Research Connects to This API

| Thesis Contribution | How It Appears in This Codebase |
|---------------------|---------------------------------|
| Three-tier cascade architecture | `/predict` (Tier 2) + async Celery escalation pattern |
| DistilBERT-FGM inference pipeline | `classifier.py` — `classify()` function |
| False Positive Rate optimisation | Decision thresholds: spam > 0.70, ham < 0.30 |
| Human-in-the-loop escalation design | Celery task queue + `/task/{id}` polling workflow |
| Multi-corpus training (4 corpora) | Model generalises across Enron, LingSpam, SpamAssassin, TREC-07 |
| FGM adversarial robustness | Model hardened against text-based adversarial perturbations |

---

## Roadmap

- [ ] Upload model checkpoint to HuggingFace Hub for fully self-contained deployment
- [ ] Replace default PyTorch wheel with `torch+cpu` to reduce Docker image size by ~450 MB
- [ ] Integrate Tier 1 (SpamAssassin) and Tier 3 (Flan-T5) cascade components
- [ ] Add `POST /feedback` endpoint for human reviewer label submission
- [ ] Replace `Base.metadata.create_all()` with Alembic versioned migrations
- [ ] Add JWT authentication for production endpoint security

---

## License

This project is released under the [MIT License](LICENSE).

---

*Bari Asunka — Chaoyang University of Technology, Taichung, Taiwan — June 2026*
