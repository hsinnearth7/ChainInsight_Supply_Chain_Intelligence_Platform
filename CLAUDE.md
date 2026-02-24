# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChainInsight — End-to-end supply chain inventory analytics platform. Supports both **batch mode** (original scripts) and **live mode** (FastAPI + React SPA + SQLite). Generates a dirty 10k-row CSV, applies an 8-step ETL cleaning pipeline, then runs statistical analysis, advanced optimization (EOQ, Monte Carlo), 30 ML algorithms, and 6 RL agents. Outputs a cleaned CSV with 3 derived fields and 28+ publication-ready charts. Bilingual project: every script has both Chinese and English versions.

## Running

### Live Mode (Phase 3 — Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

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

### Batch Mode (Original)

```bash
python generate_data.py
python "clean_data .py"
python "chart_0_Inventory Health & Stockout Risk.py.txt"
python chart_01_to_08_statistical_analysis.py.txt
python chart_09_to_14_advanced_supply_chain.py.txt
python chart_15_to_22_ai_algorithms_analysis.py.txt
```

## Architecture (Live Mode — Phase 3)

```
Upload CSV ──▸ FastAPI /api/ingest ──▸ PipelineOrchestrator (async)
  (or)                                        │
Watchdog ──▸ data/raw/ ──▸ auto-trigger       ├── on_progress callback
                                               │         │
                    ┌──────────────────────────┤         ▼
                    ▼                          ▼    WebSocket broadcast
              ETL Pipeline          SQLite DB       /ws/pipeline/{batch_id}
                    │                                    │
              ▼     ▼     ▼     ▼                       │
           Stats  SCM   ML    RL                        │
              │     │     │     │                        ▼
              ▼     ▼     ▼     ▼              React SPA (real-time)
         28 PNG charts + KPI JSON              /ws/global (watchdog)
                    │
              ▼           ▼
        React SPA     REST API /api/*
```

### Key Modules

| Module | Path | Description |
|--------|------|-------------|
| Config | `app/config.py` | Centralized settings (dirs, params, colors) |
| ETL | `app/pipeline/etl.py` | `ETLPipeline` class — 8-step cleaning |
| Stats | `app/pipeline/stats.py` | `StatisticalAnalyzer` — charts 0-8, KPIs |
| Supply Chain | `app/pipeline/supply_chain.py` | `SupplyChainAnalyzer` — charts 9-14 |
| ML Engine | `app/pipeline/ml_engine.py` | `MLAnalyzer` — charts 15-22, GA |
| Orchestrator | `app/pipeline/orchestrator.py` | `PipelineOrchestrator` — coordinates all stages |
| RL Environment | `app/rl/environment.py` | Gymnasium `InventoryEnv` — 5-state, 5-action |
| RL Agents | `app/rl/agents/*.py` | Q-Learning, SARSA, DQN, PPO, A2C, GA-RL Hybrid |
| RL Trainer | `app/rl/trainer.py` | `RLTrainer` — trains all agents, collects metrics |
| RL Evaluator | `app/rl/evaluator.py` | `RLEvaluator` — charts 23-28, KPI comparison |
| DB Models | `app/db/models.py` | SQLAlchemy: InventorySnapshot, AnalysisResult, PipelineRun |
| API Routes | `app/api/routes.py` | FastAPI REST endpoints (async ingest + 10 endpoints) |
| API Entry | `app/main.py` | FastAPI app with WS routes, watchdog, SPA mount |
| WS Manager | `app/ws/manager.py` | WebSocket ConnectionManager (global + per-batch) |
| WS Routes | `app/ws/routes.py` | `/ws/pipeline/{batch_id}`, `/ws/global` |
| File Watcher | `app/watcher.py` | Watchdog monitor for `data/raw/` with 2s debounce |
| React Frontend | `frontend/` | Vite + React 18 + TypeScript + Tailwind + Recharts |

### Data Flow

1. CSV uploaded via React UI or auto-detected by Watchdog in `data/raw/`
2. `POST /api/ingest` saves file, creates queued `PipelineRun`, returns immediately
3. `asyncio.to_thread(orchestrator.run, ...)` executes pipeline in background
4. `on_progress` callback bridges sync thread → async WS broadcast via `run_coroutine_threadsafe()`
5. React frontend receives real-time updates on `/ws/pipeline/{batch_id}`
6. Each stage saves charts to `data/charts/{batch_id}/` and results to SQLite
7. React pages fetch data via REST API and render interactive Recharts + PNG images

### WebSocket Message Protocol

```json
{
  "type": "pipeline:progress|pipeline:completed|pipeline:failed|watchdog:detected",
  "batch_id": "batch_20240101_120000_abc123",
  "timestamp": "2024-01-01T12:00:00Z",
  "payload": {
    "stage": "etl|stats|supply_chain|ml|rl",
    "status": "running|completed",
    "progress_pct": 0-100,
    "data": {}
  }
}
```

### REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/ingest` | Upload CSV, trigger async pipeline |
| GET | `/api/runs` | List all runs (latest 50) |
| GET | `/api/runs/{batch_id}` | Full run details + analyses |
| GET | `/api/runs/{batch_id}/status` | Quick status poll (WS fallback) |
| GET | `/api/runs/{batch_id}/kpis` | Stats KPIs for a run |
| GET | `/api/runs/{batch_id}/analysis/{type}` | KPIs + chart_paths for a stage |
| GET | `/api/runs/{batch_id}/data` | Inventory snapshot rows |
| GET | `/api/runs/{batch_id}/charts` | List chart files |
| GET | `/api/runs/{batch_id}/charts/{name}` | Serve chart PNG |
| GET | `/api/latest/kpis` | Most recent completed KPIs |
| GET | `/api/history/kpis` | KPI trend history |

### React Frontend Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | DashboardPage | 5 KPI cards, stock status pie, category bar, vendor table, alerts |
| `/upload` | UploadPage | File uploader, real-time pipeline progress via WS |
| `/stats` | StatsPage | 3 tabs: Interactive charts / PNG charts / Raw data |
| `/supply-chain` | SupplyChainPage | EOQ calculator, Monte Carlo simulator, PNG charts 9-14 |
| `/ml` | MLPage | Algorithm summary table, PNG charts 15-22 |
| `/rl` | RLPage | Agent KPIs, comparison table, interactive reward curves, PNG charts 23-28 |
| `/history` | HistoryPage | Run table with status, KPI trend chart |

### i18n (Internationalization)

- **Languages**: English (en), Chinese (zh), Japanese (ja)
- **Translation file**: `frontend/src/i18n/translations.ts` — all UI strings as `{ en, zh, ja }` objects
- **Hook**: `frontend/src/i18n/useTranslation.ts` — `useTranslation()` returns `{ t, language }`
- **State**: Zustand `appStore.language` with `localStorage` persistence (`ci-lang` key)
- **Switcher**: Header component — 3-button toggle (EN / 中文 / 日本語)

### Database Tables

- `inventory_snapshots` — one row per product per batch (time-series ready)
- `analysis_results` — KPI JSON + chart paths per stage per batch
- `pipeline_runs` — batch metadata, status, timing

## File Structure

### Original Scripts (Batch Mode)
| File | Description |
|------|-------------|
| `generate_data.py` | Synthetic dirty data generator |
| `clean_data .py` / `clean_data_en.py.txt` | ETL pipeline (CN / EN) |
| `chart_0_*.py.txt` / `.en.py.txt` | Inventory health dashboard |
| `chart_01_to_08_*.py.txt` / `_en.py.txt` | Statistical analysis |
| `chart_09_to_14_*.py.txt` / `_en.py.txt` | Advanced optimization |
| `chart_15_to_22_*.py.txt` / `_en.py.txt` | 30 AI algorithms |

### Live Mode (Phase 1 + 2 + 3)
| File | Description |
|------|-------------|
| `app/main.py` | FastAPI entry point (WS, watchdog, SPA) |
| `app/config.py` | Configuration |
| `app/pipeline/*.py` | Modular pipeline stages (ETL, Stats, SCM, ML) |
| `app/rl/environment.py` | Gymnasium InventoryEnv |
| `app/rl/agents/*.py` | 6 RL agents |
| `app/rl/trainer.py` | RL training coordinator |
| `app/rl/evaluator.py` | RL chart generator (charts 23-28) |
| `app/db/models.py` | SQLAlchemy models |
| `app/api/routes.py` | REST API routes (async) |
| `app/ws/manager.py` | WebSocket ConnectionManager |
| `app/ws/routes.py` | WebSocket route handlers |
| `app/watcher.py` | Watchdog file monitor |
| `frontend/` | React SPA (Vite + TypeScript + Tailwind + Recharts + Zustand + i18n) |

## ETL Pipeline

**8 steps** (identical logic in both batch and live mode):
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

**Categories (7):** Electronics, Home, Food, Shipping, Office, Apparel, Industrial
**Vendors (7):** Tokyo Electronics, Fukuoka Logistics, Hokkaido Foods, Kyoto Crafts, Osaka Supplies, Nagoya Parts, Sapporo Steel

## Dependencies

Core: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `squarify`
Live: `fastapi`, `uvicorn`, `sqlalchemy`, `websockets`, `watchdog`
RL: `gymnasium`, `torch`
Frontend: `react`, `react-router-dom`, `recharts`, `zustand`, `tailwindcss`, `vite`
Optional: `xgboost`, `streamlit`, `plotly`

## Roadmap

- [x] Phase 1: Modular pipeline + FastAPI + Streamlit + SQLite
- [x] Phase 2: RL engine (Gymnasium + Q-Learning/SARSA/DQN/PPO/A2C/GA-RL Hybrid)
- [x] Phase 3: WebSocket real-time + React frontend + Watchdog file monitoring
- [ ] Phase 4: PostgreSQL + Docker + LSTM/Transformer demand forecasting
