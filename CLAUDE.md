# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChainInsight — End-to-end supply chain inventory analytics platform with **live mode** (FastAPI + React SPA + SQLite). Applies an 8-step ETL cleaning pipeline, runs statistical analysis, advanced optimization (EOQ, Monte Carlo), 30 ML algorithms, and 6 RL agents. Outputs a cleaned CSV with 3 derived fields and 28+ publication-ready charts.

## Running

### Live Mode (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and set API_KEY for production

# React Frontend (dev mode)
cd frontend && npm install && npm run dev
# Opens at http://localhost:5173 (proxies API to :8000)

# FastAPI Backend
uvicorn app.main:app --reload --port 8000
# API docs at http://localhost:8000/docs

# Production (single server)
cd frontend && npm run build
uvicorn app.main:app --port 8000
# Serves both API and React UI at http://localhost:8000
```

### Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check app/ tests/
```

## Authentication

All API endpoints require an `X-API-Key` header (except `/api/health`).
WebSocket connections require `?api_key=...` query parameter.
Configure via `API_KEY` env var (defaults to `dev-key-change-me` in dev).

## Architecture

```
Upload CSV ──▸ FastAPI /api/ingest ──▸ PipelineOrchestrator (async)
  (or)           (API Key auth)                │
Watchdog ──▸ data/raw/ ──▸ auto-trigger       ├── on_progress callback
                                               │         │
                    ┌──────────────────────────┤         ▼
                    ▼                          ▼    WebSocket broadcast
              ETL Pipeline          SQLite DB       /ws/pipeline/{batch_id}
                    │             (FK relations)          │
              ▼     ▼     ▼     ▼                        │
           Stats  SCM   ML    RL                         │
              │     │     │     │                         ▼
              ▼     ▼     ▼     ▼               React SPA (real-time)
         28 PNG charts + KPI JSON               /ws/global (watchdog)
                    │
              ▼           ▼
        React SPA     REST API /api/*
```

### Key Modules

| Module | Path | Description |
|--------|------|-------------|
| Config | `app/config.py` | Settings, enums (`PipelineStatus`, `StockStatus`), constants |
| Auth | `app/auth.py` | API Key authentication (`require_api_key` dependency) |
| Enrichment | `app/pipeline/enrichment.py` | Shared `enrich_base()` (DSI, Coverage, Demand Intensity) |
| ETL | `app/pipeline/etl.py` | `ETLPipeline` — 8-step cleaning with schema validation |
| Stats | `app/pipeline/stats.py` | `StatisticalAnalyzer` — charts 0-8, KPIs |
| Supply Chain | `app/pipeline/supply_chain.py` | `SupplyChainAnalyzer` — charts 9-14 |
| ML Engine | `app/pipeline/ml_engine.py` | `MLAnalyzer` — charts 15-22, no data leakage |
| Orchestrator | `app/pipeline/orchestrator.py` | `PipelineOrchestrator` — coordinates all stages |
| Service Layer | `app/services/pipeline_service.py` | Pipeline run management |
| RL Base | `app/rl/agents/base.py` | `BaseTabularAgent`, `BasePolicyAgent` ABCs |
| RL Environment | `app/rl/environment.py` | Gymnasium `InventoryEnv` — 5-state, 5-action |
| RL Agents | `app/rl/agents/*.py` | Q-Learning, SARSA, DQN, PPO, A2C, GA-RL Hybrid |
| RL Trainer | `app/rl/trainer.py` | `RLTrainer` — seed control, convergence detection |
| RL Evaluator | `app/rl/evaluator.py` | `RLEvaluator` — charts 23-28, KPI comparison |
| DB Models | `app/db/models.py` | SQLAlchemy with FK relationships (PipelineRun → Snapshots/Results) |
| API Routes | `app/api/routes.py` | FastAPI REST endpoints (auth, rate limiting, path validation) |
| API Entry | `app/main.py` | FastAPI app with CORS, WS routes, watchdog, SPA mount |
| WS Manager | `app/ws/manager.py` | WebSocket ConnectionManager (global + per-batch) |
| WS Routes | `app/ws/routes.py` | `/ws/pipeline/{batch_id}`, `/ws/global` (API key required) |
| File Watcher | `app/watcher.py` | Watchdog monitor for `data/raw/` with 2s debounce |
| React Frontend | `frontend/` | Vite + React 18 + TypeScript + Tailwind + Recharts |

### Security

- **Authentication**: API Key via `X-API-Key` header (all endpoints except `/api/health`)
- **CORS**: Configurable origins via `CORS_ORIGINS` env var (not `*`)
- **Path Traversal**: `_safe_path()` validates all file paths stay within base directory
- **Upload Security**: Size limits, filename sanitization, CSV validation
- **Rate Limiting**: In-memory per-IP rate limiter

### ML Pipeline

- **No data leakage**: `StandardScaler` wrapped in `sklearn.pipeline.Pipeline` for CV
- **Feature separation**: `CLASSIFICATION_FEATURES` (no circular predictors) vs `REGRESSION_FEATURES`
- **Enrichment**: Shared `enrich_base()` used by stats, supply_chain, and ml_engine

### RL Pipeline

- **Seed control**: Reproducible training via `RLTrainer(seed=42)`
- **Base classes**: `BaseTabularAgent` (Q-Learning, SARSA) and `BasePolicyAgent` (PPO, A2C)
- **Convergence**: Rolling window detection in `_find_convergence()`

### Data Flow

1. CSV uploaded via React UI or auto-detected by Watchdog in `data/raw/`
2. `POST /api/ingest` validates & saves file, creates queued `PipelineRun`, returns immediately
3. `asyncio.to_thread(orchestrator.run, ...)` executes pipeline in background
4. `on_progress` callback bridges sync thread → async WS broadcast
5. React frontend receives real-time updates on `/ws/pipeline/{batch_id}`
6. Each stage saves charts to `data/charts/{batch_id}/` and results to SQLite
7. React pages fetch data via REST API and render interactive Recharts + PNG images

### REST API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/health` | No | Health check |
| POST | `/api/ingest` | Yes | Upload CSV, trigger async pipeline |
| POST | `/api/ingest/existing` | Yes | Trigger pipeline with existing file |
| GET | `/api/runs` | Yes | List all runs (latest 50) |
| GET | `/api/runs/{batch_id}` | Yes | Full run details + analyses |
| GET | `/api/runs/{batch_id}/status` | Yes | Quick status poll |
| GET | `/api/runs/{batch_id}/kpis` | Yes | Stats KPIs for a run |
| GET | `/api/runs/{batch_id}/analysis/{type}` | Yes | KPIs + chart_paths for a stage |
| GET | `/api/runs/{batch_id}/data` | Yes | Inventory snapshot rows |
| GET | `/api/runs/{batch_id}/charts` | Yes | List chart files |
| GET | `/api/runs/{batch_id}/charts/{name}` | Yes | Serve chart PNG |
| GET | `/api/latest/kpis` | Yes | Most recent completed KPIs |
| GET | `/api/history/kpis` | Yes | KPI trend history |

### Database Tables

- `pipeline_runs` — batch metadata, status, timing (PK: batch_id)
- `inventory_snapshots` — one row per product per batch (FK → pipeline_runs.batch_id)
- `analysis_results` — KPI JSON + chart paths per stage per batch (FK → pipeline_runs.batch_id)

## Configuration

All configuration via environment variables (see `.env.example`):
- `API_KEY` — Authentication key
- `CORS_ORIGINS` — Comma-separated allowed origins
- `DATABASE_URL` — SQLAlchemy connection string
- `MAX_UPLOAD_MB` — Upload size limit (default: 10)
- `RATE_LIMIT_PER_MINUTE` — Rate limit (default: 30)

Named constants and enums defined in `app/config.py`:
- `PipelineStatus` / `StockStatus` enums
- `DSI_SENTINEL`, `ABC_THRESHOLD_A/B`, `SUPPLY_RISK_WEIGHTS`, etc.

## ETL Pipeline

**8 steps** with schema validation:
1. `Product_ID` — strip whitespace
2. `Category` — strip + capitalize
3. `Unit_Cost_Raw` → `Unit_Cost` — regex extraction
4. `Current_Stock_Raw` → `Current_Stock` — coerce + clamp negatives
5. Null handling — stock→0, cost→category median
6. `Vendor_Name` — strip whitespace
7. Validation — clip to valid ranges
8. Derived: `Reorder_Point`, `Stock_Status`, `Inventory_Value`

## Data Schema

**Input (8 cols):** Product_ID, Category, Unit_Cost_Raw, Current_Stock_Raw, Daily_Demand_Est, Safety_Stock_Target, Vendor_Name, Lead_Time_Days

**Output (11 cols):** Product_ID, Category, Unit_Cost, Current_Stock, Daily_Demand_Est, Safety_Stock_Target, Vendor_Name, Lead_Time_Days, Reorder_Point, Stock_Status, Inventory_Value

## Dependencies

Core: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `squarify`
Live: `fastapi`, `uvicorn`, `sqlalchemy`, `websockets`, `watchdog`
RL: `gymnasium`, `torch`
Frontend: `react`, `react-router-dom`, `recharts`, `zustand`, `tailwindcss`, `vite`
Dev: `pytest`, `pytest-asyncio`, `httpx`, `ruff`, `mypy`
