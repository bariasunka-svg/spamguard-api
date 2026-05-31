# SpamGuard API

> A production-grade spam detection REST API built on a DistilBERT-FGM model trained across four benchmark email corpora as part of a Master's thesis in Information Management at Chaoyang University of Technology (CYUT), Taiwan.

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

---

## Overview

SpamGuard API exposes a trained DistilBERT-FGM email classification model as a fully containerised REST service. The service accepts raw email text and returns a structured spam verdict with a calibrated confidence score, classifying messages as `spam`, `ham` (legitimate), or `uncertain`.

The system implements the Tier 2 component of a three-tier cascade detection architecture described in the author's Master's thesis. Emails classified as `uncertain` by Tier 2 are architecturally intended to escalate to a Flan-T5 large language model (Tier 3) for human-in-the-loop review — a workflow supported by the asynchronous batch processing endpoints included in this release.

**Key characteristics:**

- Sub-200ms inference latency on CPU-only deployment targets
- Persistent classification audit log via SQLAlchemy ORM
- Asynchronous batch processing via Celery task queue with Redis broker
- Full OpenAPI documentation auto-generated at `/docs`
- Three-container deployment orchestrated with a single `docker compose up` command

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

| Tier | Component | Trigger Condition | Role |
|------|-----------|-------------------|------|
| 1 | SpamAssassin (rule-based) | Score > 8.0 or < 2.0 | Fast triage for obvious cases |
| **2** | **DistilBERT-FGM** *(this API)* | **Confidence ∈ [0.3, 0.7] → uncertain** | **Primary AI classifier** |
| 3 | Flan-T5-base (LLM) | Tier 2 returns `uncertain` | Human-in-the-loop escalation |

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

### Why This Model Was Selected

The thesis conducted a systematic experiment training 30 model configurations (2 training methods × 3 token lengths × 5 random seeds) and evaluating each under Leave-One-Corpus-Out (LOCO) cross-domain testing. The primary optimisation target was **False Positive Rate (FPR)** — the probability of a legitimate email being incorrectly classified as spam — as this represents the highest-cost error in a production email security system.

The `p3_full_fgm_ml256_s456` configuration achieved the lowest cross-domain FPR across all evaluation conditions, demonstrating robust generalisation to unseen email distributions.

**FGM adversarial training** exposes the model to adversarially perturbed training examples, improving resistance to subtle text variations characteristic of modern spam campaigns — a property directly relevant to cybersecurity deployment contexts.

### Inference Decision Thresholds

```
p_spam > 0.70  →  verdict: "spam"    (high confidence — block)
p_spam < 0.30  →  verdict: "ham"     (high confidence — deliver)
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
- The trained model checkpoint placed at `./models/p3_full_fgm_ml256_s456/best/`  
  *(Required files: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`)*

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

This command starts three containers: the FastAPI application server, the Celery worker, and the Redis message broker. The DistilBERT model loads into memory during startup (~30 seconds on first run).

### 4. Access the API

| Interface | URL |
|-----------|-----|
| Interactive API documentation (Swagger UI) | http://localhost:8000/docs |
| OpenAPI schema | http://localhost:8000/openapi.json |
| Health check | http://localhost:8000/health |

### 5. Run a quick test

```bash
# Spam classification
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Congratulations! You have won $1,000,000. Click here now!", "email_id": "test001"}'

# Expected response
{
  "email_id": "test001",
  "result": "spam",
  "confidence": 0.9981,
  "tier_used": 2
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
Dispatches a batch of emails for asynchronous classification via the Celery task queue. Returns immediately with a task identifier.

**Request body**
```json
{
  "emails": ["email body 1", "email body 2", "..."]
}
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

**Response (pending)**
```json
{"task_id": "string", "status": "PENDING", "result": null}
```

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

This project constitutes the deployment layer of a Master's thesis submitted in partial fulfilment of the requirements for the degree of Master of Science in Information Management at Chaoyang University of Technology (CYUT), Taiwan.

**Thesis title:** *Cross-Domain Email Spam Detection Using Adversarially Trained Transformer Models*

**Author:** Bari Asunka

**Institution:** Chaoyang University of Technology (CYUT), Taichung, Taiwan

### Thesis–API Correspondence

| Thesis Component | API Implementation |
|------------------|--------------------|
| Three-tier cascade architecture | `/predict` (Tier 2) + `/predict/batch` async escalation pattern |
| DistilBERT-FGM inference pipeline | `classifier.py` — `classify()` function |
| False Positive Rate optimisation | Decision thresholds: spam > 0.70, ham < 0.30 |
| Human-in-the-loop escalation | Celery task queue + `/task/{id}` polling |
| Cross-domain robustness | Multi-corpus training (Enron, LingSpam, SpamAssassin, TREC-07) |
| Adversarial robustness (FGM) | Model hardened against text-based adversarial perturbations |

---

## Roadmap

- [ ] Upload model checkpoint to HuggingFace Hub for one-command deployment
- [ ] Replace CPU PyTorch wheel with `torch+cpu` to reduce Docker image size by ~450 MB
- [ ] Integrate Tier 1 (SpamAssassin) and Tier 3 (Flan-T5) cascade components
- [ ] Add `POST /feedback` endpoint for human reviewer label submission
- [ ] Replace `Base.metadata.create_all()` with Alembic versioned migrations
- [ ] Add JWT authentication for production endpoint security

---

## License

This project is released under the [MIT License](LICENSE).

---

*Developed as part of a backend engineering internship preparation sprint — June 2026*  
*Chaoyang University of Technology, Taichung, Taiwan*
