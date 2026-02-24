<div align="center">

# ChainInsight - Supply Chain Intelligence Platform

**End-to-End Supply Chain Inventory Analytics: From Dirty Data to Business Decisions**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/pandas-1.5+-150458.svg)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Trilingual](https://img.shields.io/badge/lang-EN%20%7C%20中文%20%7C%20日本語-red.svg)](#language-support)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

<br>

<img src="https://img.shields.io/badge/Data%20Pipeline-ETL-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Analysis-Statistical%20%2B%20ML-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/RL%20Agents-6-red?style=for-the-badge" />
<img src="https://img.shields.io/badge/Rows-10%2C000%2B-green?style=for-the-badge" />
<img src="https://img.shields.io/badge/Algorithms-30-purple?style=for-the-badge" />

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [React SPA (Phase 3)](#react-spa-phase-3)
- [Interactive Dashboard](#interactive-dashboard)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Data Schema](#data-schema)
- [Getting Started](#getting-started)
- [Pipeline Details](#pipeline-details)
  - [Phase 1: Data Generation](#phase-1-data-generation)
  - [Phase 2: Data Cleaning & ETL](#phase-2-data-cleaning--etl)
  - [Phase 3: Inventory Health Analysis](#phase-3-inventory-health-analysis)
  - [Phase 4: Statistical Analysis](#phase-4-statistical-analysis)
  - [Phase 5: Advanced Supply Chain Optimization](#phase-5-advanced-supply-chain-optimization)
  - [Phase 6: AI & Machine Learning](#phase-6-ai--machine-learning)
- [Visualization Gallery](#visualization-gallery)
- [Business Insights & Findings](#business-insights--findings)
- [Roadmap & Future Work](#roadmap--future-work)
- [Tech Stack](#tech-stack)
- [Language Support](#language-support)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

**ChainInsight** is a comprehensive, end-to-end supply chain inventory analytics platform that demonstrates the complete data science lifecycle — from synthetic dirty data generation, through robust ETL cleaning, to advanced statistical modeling, 30 machine learning algorithms, and 6 reinforcement learning agents. The platform features a modern **React 18 + TypeScript** SPA with **WebSocket real-time** pipeline monitoring, powered by a **FastAPI** backend with SQLite persistence.

This project simulates a real-world supply chain scenario: a 10,000+ row inventory dataset plagued with missing values, inconsistent formats, and data quality issues. The pipeline cleans, transforms, analyzes, and visualizes this data, ultimately delivering actionable business intelligence for inventory optimization — viewable through an interactive trilingual dashboard (EN / 中文 / 日本語).

### Why This Project?

| Challenge | Our Approach |
|-----------|-------------|
| Real-world data is messy | Generate realistic dirty data with 1,280+ quality issues |
| ETL pipelines lack transparency | 8-step documented transformation with before/after tracking |
| Analysis often stops at EDA | Full pipeline from descriptive stats → ML → RL optimization |
| Supply chain decisions need data | Actionable KPIs: EOQ, Reorder Points, Stockout Probability |
| No real-time visibility | WebSocket live progress + Watchdog auto-trigger on CSV drop |
| Language barrier | Trilingual React UI (EN / 中文 / 日本語) + bilingual batch scripts |

---

## Key Features

- **Synthetic Data Generator** — Configurable dirty data engine that injects realistic quality issues (mixed formats, negatives, nulls, duplicates, encoding inconsistencies)
- **8-Step ETL Pipeline** — Robust data cleaning using pandas & numpy with regex parsing, category-median imputation, and business-logic validation
- **Inventory Health Dashboard** — DSI (Days Sales of Inventory), ABC Pareto classification, stock status distribution, and stockout alert reporting
- **Statistical Analysis Suite** — Pearson/Spearman correlation, Chi-square independence tests, ANOVA/Kruskal-Wallis group comparisons, outlier detection, and supply risk scoring
- **Advanced Optimization** — EOQ (Economic Order Quantity) modeling, Monte Carlo stockout simulation, and inventory cost optimization
- **30 ML Algorithms** — Classification, regression, clustering, dimensionality reduction, anomaly detection, and genetic algorithm optimization applied to supply chain data
- **28 Publication-Ready Charts** — Comprehensive matplotlib/seaborn visualizations covering every analysis phase
- **6 Reinforcement Learning Agents** — Q-Learning, SARSA, DQN, PPO, A2C, GA-RL Hybrid for dynamic inventory control
- **React SPA + WebSocket Real-Time** — Modern React 18 + TypeScript frontend with live pipeline progress via WebSocket
- **Trilingual UI (EN / 中文 / 日本語)** — Full i18n language switching with `localStorage` persistence
- **Watchdog Auto-Trigger** — File monitoring auto-detects CSV drops in `data/raw/` and triggers pipeline
- **Bilingual Codebase** — Every script has both Chinese and English versions with identical functionality

---

## Interactive Dashboard

ChainInsight includes a **single-file vanilla HTML dashboard** (`dashboard.html`) that showcases all analytics results in a polished, dark-themed interface with **trilingual i18n support (中文 / EN / 日本語)**.

### Features

- **6 Dashboard Views** — Overview, Data Pipeline, Statistical Analysis, Supply Chain Optimization, AI/ML Algorithms, Inventory Health
- **Live KPI Computation** — Reads `Supply_Chain_Inventory_Clean.csv` and computes KPIs (Total Products, Categories, Vendors, Avg Unit Cost, OOS Rate, Total Inventory Value) client-side
- **Interactive Charts** — SVG pie, bar, and donut charts rendered from CSV data; pre-generated PNG chart gallery with lightbox zoom
- **Language Switching** — Toggle between 中文 / EN / 日本語 with all labels, titles, and descriptions fully translated; persisted in `localStorage`
- **30-Algorithm Summary Table** — Complete overview of all ML algorithms with status, category, and supply chain use case
- **ETL Pipeline Visualization** — Visual 8-step pipeline walkthrough with before/after statistics

### How to Launch

```bash
# Option 1: Open directly in browser (charts won't load due to CORS)
# Option 2: Serve via local HTTP server (recommended)
cd ChainInsight
python -m http.server 8080
# Then open http://localhost:8080/dashboard.html
```

> **Note:** A local HTTP server is required for the CSV data fetch and chart image loading to work correctly.

---

## React SPA (Phase 3)

ChainInsight includes a modern **React 18 + TypeScript** single-page application with real-time pipeline monitoring.

### Features

- **7 Pages** — Dashboard, Upload & Run, Statistics, Supply Chain, ML/AI, RL Optimization, History
- **WebSocket Real-Time** — Live pipeline progress updates via WebSocket (`/ws/pipeline/{batch_id}`)
- **Interactive Charts** — Recharts-powered EOQ calculator, Monte Carlo simulator, reward learning curves, correlation heatmaps
- **Trilingual i18n** — Toggle between **EN / 中文 / 日本語** with all UI labels fully translated; language persisted in `localStorage`
- **Dark Mode** — One-click theme toggle with Tailwind CSS `class` strategy
- **Watchdog Notifications** — Real-time alerts when CSV files are auto-detected in `data/raw/`
- **28 PNG Charts + Interactive Charts** — Both server-generated static charts and client-side interactive visualizations

### How to Launch

```bash
# Development mode (hot reload)
cd frontend && npm install && npm run dev
# Opens at http://localhost:5173 (proxies API to :8000)
# In a separate terminal:
uvicorn app.main:app --reload --port 8000

# Production mode (single server)
cd frontend && npm run build
uvicorn app.main:app --port 8000
# Serves both API and React UI at http://localhost:8000
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 18 + TypeScript |
| Build | Vite 6 |
| Styling | Tailwind CSS (dark mode) |
| Charts | Recharts |
| State | Zustand |
| Routing | React Router v6 |
| Real-Time | WebSocket (auto-reconnect) |
| i18n | Custom hook with 3 languages |

---

## Project Architecture

### Live Mode (Phase 3)

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
         28 PNG charts + KPI JSON              7 pages, i18n (EN/ZH/JA)
                    │
              ▼           ▼
        React SPA     REST API /api/*
```

### Batch Mode (Original)

```
  ┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────────┐
  │   GENERATE   │────>│   CLEAN/ETL   │────>│   ANALYZE    │────>│  VISUALIZE   │
  │              │     │               │     │              │     │              │
  │ generate_    │     │ clean_data.py │     │ Statistical  │     │ 28 Charts    │
  │ data.py      │     │ (8 Steps)     │     │ ML (30 algo) │     │ (matplotlib) │
  │              │     │               │     │ RL (6 agents)│     │              │
  │ 10,050 rows  │     │ 8 cols → 11   │     │ Monte Carlo  │     │ PNG outputs  │
  │ 8 columns    │     │ cols          │     │              │     │              │
  └──────────────┘     └───────────────┘     └──────────────┘     └──────────────┘
        │                     │                     │                     │
        ▼                     ▼                     ▼                     ▼
  Dirty CSV            Clean CSV             Business KPIs         Visual Reports
  (1,280+ issues)      (validated)           & Predictions         & Dashboards
```

---

## Project Structure

```
ChainInsight/
│
├── README.md                                           # This file
├── CLAUDE.md                                           # Claude Code project context
├── LICENSE                                             # MIT License
├── requirements.txt                                    # Python dependencies
├── dashboard.html                                      # Static HTML dashboard (i18n: EN/JA/ZH)
│
├── app/                                                # FastAPI backend (Live Mode)
│   ├── main.py                                         # FastAPI app entry (WS, watchdog, SPA)
│   ├── config.py                                       # Centralized settings
│   ├── api/routes.py                                   # REST API endpoints (async ingest)
│   ├── ws/manager.py                                   # WebSocket ConnectionManager
│   ├── ws/routes.py                                    # WebSocket route handlers
│   ├── watcher.py                                      # Watchdog file monitor (data/raw/)
│   ├── db/models.py                                    # SQLAlchemy models
│   ├── pipeline/                                       # Pipeline stages
│   │   ├── orchestrator.py                             # Coordinates all 5 stages
│   │   ├── etl.py                                      # 8-step ETL cleaning
│   │   ├── stats.py                                    # Charts 0-8, statistical KPIs
│   │   ├── supply_chain.py                             # Charts 9-14, EOQ/Monte Carlo
│   │   └── ml_engine.py                                # Charts 15-22, 30 ML algorithms
│   └── rl/                                             # Reinforcement learning
│       ├── environment.py                              # Gymnasium InventoryEnv
│       ├── trainer.py                                  # Trains all 6 agents
│       ├── evaluator.py                                # Charts 23-28, agent comparison
│       └── agents/                                     # Q-Learning, SARSA, DQN, PPO, A2C, GA-RL
│
├── frontend/                                           # React SPA (Phase 3)
│   ├── package.json                                    # React 18, Recharts, Zustand, Tailwind
│   ├── vite.config.ts                                  # Vite proxy (API, WS, charts)
│   ├── tailwind.config.js                              # ChainInsight color palette + dark mode
│   └── src/
│       ├── App.tsx                                     # Router + global WS listener
│       ├── api/client.ts                               # Fetch-based API client
│       ├── i18n/                                       # Trilingual i18n (EN/ZH/JA)
│       │   ├── translations.ts                         # All UI strings in 3 languages
│       │   └── useTranslation.ts                       # Translation hook
│       ├── stores/appStore.ts                          # Zustand (dark mode, language, batch ID)
│       ├── hooks/                                      # useWebSocket, usePipelineProgress
│       ├── components/                                 # Sidebar, Header, KPICard, DataTable, etc.
│       └── pages/                                      # 7 page components
│
├── generate_data.py                                    # Synthetic dirty data generator
├── clean_data .py                                      # ETL pipeline (Chinese)
├── chart_*_*.py.txt                                    # Analysis scripts (CN & EN versions)
│
├── data/                                               # Runtime data directory
│   ├── raw/                                            # Watchdog-monitored CSV drop zone
│   └── charts/{batch_id}/                              # Generated charts per batch
│
└── chaininsight.db                                     # SQLite database
```

---

## Data Schema

### Input: Raw Dirty Data (8 columns)

| Column | Type | Dirty Issues |
|--------|------|-------------|
| `Product_ID` | string | Leading/trailing whitespace |
| `Category` | string | Inconsistent casing (`electronics` vs `Electronics` vs `ELECTRONICS`) |
| `Unit_Cost_Raw` | string | Mixed formats: `"USD 45.99"`, `"$30"`, `"Quote Pending"`, plain numbers |
| `Current_Stock_Raw` | mixed | Negative values, nulls, non-numeric entries (e.g. `"120 pcs"`) |
| `Daily_Demand_Est` | float | 5% missing values |
| `Safety_Stock_Target` | float | Occasional negatives |
| `Vendor_Name` | string | Whitespace padding |
| `Lead_Time_Days` | int | Values < 1 |

### Output: Clean Data (11 columns)

| Column | Type | Transformation |
|--------|------|---------------|
| `Product_ID` | string | Stripped whitespace |
| `Category` | string | Standardized capitalization |
| `Unit_Cost` | float | Regex-extracted numeric, imputed nulls |
| `Current_Stock` | float | Coerced numeric, negatives → 0 |
| `Daily_Demand_Est` | float | Clipped to >= 0 |
| `Safety_Stock_Target` | float | Clipped to >= 0 |
| `Vendor_Name` | string | Stripped whitespace |
| `Lead_Time_Days` | float | Clipped to >= 1 |
| **`Reorder_Point`** | float | *Derived:* Demand × Lead Time + Safety Stock |
| **`Stock_Status`** | string | *Derived:* Out of Stock / Low Stock / Normal Stock |
| **`Inventory_Value`** | float | *Derived:* Current Stock × Unit Cost |

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 18+ (for React frontend)
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/hsinnearth7/ChainInsight_Supply_Chain_Intelligence_Platform.git
cd ChainInsight_Supply_Chain_Intelligence_Platform

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies & build
cd frontend && npm install && npm run build && cd ..
```

### Quick Start — Live Mode (Recommended)

```bash
# Start the server (serves both API and React UI)
uvicorn app.main:app --port 8000

# Open http://localhost:8000 in your browser
# Upload a CSV or use the existing dirty data to run the full pipeline
```

### Quick Start — Batch Mode (Original Scripts)

```bash
# Step 1: Generate synthetic dirty data (10,050 rows)
python generate_data.py

# Step 2: Run ETL cleaning pipeline
python "clean_data .py"          # Chinese version

# Step 3: Run analysis modules
python "chart_0_Inventory Health & Stockout Risk.py.txt"
python chart_01_to_08_statistical_analysis.py.txt
python chart_09_to_14_advanced_supply_chain.py.txt
python chart_15_to_22_ai_algorithms_analysis.py.txt
```

---

## Pipeline Details

### Phase 1: Data Generation

> `generate_data.py` — Synthetic dirty data engine

The generator creates a realistic 10,050-row inventory dataset with intentionally injected data quality issues that mirror real-world supply chain data problems:

```python
# Data quality issues injected:
- Mixed currency formats    → "USD 45.99", "$30", "1,200.00", "Quote Pending"
- Negative inventory        → Current_Stock = -15 (physically impossible)
- Non-numeric stock entries → "120 pcs" (unit suffix contamination)
- Whitespace contamination  → " SKU-A1234 ", leading/trailing spaces
- Case inconsistencies      → "electronics", "Electronics", "ELECTRONICS"
- Missing values            → ~5% nulls in demand estimates, ~5% nulls in cost
- Duplicate records         → 50 intentional duplicate rows
```

**Categories (7):** Electronics, Home, Food, Shipping, Office, Apparel, Industrial
**Vendors (7):** Tokyo Electronics, Fukuoka Logistics, Hokkaido Foods, Kyoto Crafts, Osaka Supplies, Nagoya Parts, Sapporo Steel
**Output:** `Supply_Chain_Inventory_Dirty_10k.csv`

---

### Phase 2: Data Cleaning & ETL

> `clean_data .py` — 8-step transformation pipeline

The core ETL pipeline applies systematic transformations using pandas and numpy:

| Step | Operation | Technique |
|------|-----------|-----------|
| 1 | Product ID cleanup | `str.strip()` |
| 2 | Category standardization | `str.strip().str.capitalize()` — deduplicates variant spellings |
| 3 | Cost extraction | `re.sub(r'[^\d.]', '', val)` — regex parsing of mixed formats |
| 4 | Stock normalization | `pd.to_numeric(errors='coerce')` + negatives → 0 |
| 5 | Null imputation | Two-pass strategy: category median → global median fallback |
| 6 | Vendor cleanup | `str.strip()` |
| 7 | Range validation | `.clip()` with business-logic bounds (demand >= 0, lead time >= 1) |
| 8 | Derived fields | Reorder Point, Stock Status classification, Inventory Value |

**Key Data Cleaning Techniques:**

```python
# Regex-based cost extraction from mixed formats
def clean_cost(value):
    if not any(char.isdigit() for char in str(value)):
        return np.nan
    cleaned = re.sub(r'[^\d.]', '', str(value))
    return float(cleaned) if cleaned else np.nan

# Two-pass null imputation strategy
df['Unit_Cost'] = df.groupby('Category')['Unit_Cost'].transform(
    lambda x: x.fillna(x.median())
)
df['Unit_Cost'] = df['Unit_Cost'].fillna(df['Unit_Cost'].median())

# Business-logic stock status classification
df['Stock_Status'] = df.apply(
    lambda row: 'Out of Stock' if row['Current_Stock'] == 0
    else 'Low Stock' if row['Current_Stock'] < row['Reorder_Point']
    else 'Normal Stock', axis=1
)
```

**Output:** `Supply_Chain_Inventory_Clean.csv` (11 columns, validated)

---

### Phase 3: Inventory Health Analysis

> `chart_0_Inventory Health & Stockout Risk.py.txt` — Operational KPIs & ABC Classification

| Metric | Formula | Business Purpose |
|--------|---------|-----------------|
| **DSI** | Current_Stock / Daily_Demand_Est | Days of inventory on hand |
| **Days to Deplete** | Current_Stock / Daily_Demand_Est | Urgency indicator for replenishment |
| **Suggested Reorder Qty** | Reorder_Point - Current_Stock (clipped >= 0) | Suggested order amount |
| **ABC Classification** | Pareto-based (cumulative inventory value: A <= 80%, B <= 95%, C > 95%) | Prioritize high-value items |

**Dashboard Panels:**
- KPI Cards — Inventory Turnover, Avg DSI, OOS Rate, Slow-Moving Value, Total Inventory Value
- Inventory Efficiency Matrix — DSI vs. Inventory Value scatter with ABC color coding
- Pareto Chart — ABC analysis by category and by individual SKU (Top 30)
- Stockout Alert Table — Top 20 at-risk SKUs where Days to Deplete < Lead Time

---

### Phase 4: Statistical Analysis

> `chart_01_to_08_statistical_analysis.py.txt` — Hypothesis testing & correlation discovery

**Correlation Analysis (chart_01):**
- Pearson correlation coefficient (linear relationships)
- Spearman rank correlation (monotonic relationships)
- P-value significance testing for all 10 variable pairs

**Distribution Analysis (chart_02):**
- Histogram + KDE for 6 key metrics
- Q-Q plots for normality assessment
- Shapiro-Wilk test with skewness/kurtosis statistics

**Hypothesis Testing (chart_03, chart_04):**
- **ANOVA (F-test)** — Vendor performance differences across metrics
- **Kruskal-Wallis test** — Non-parametric group comparison
- **Chi-square test** — Independence between Category and Stock Status
- **Category x Vendor heatmaps** — Cross-tabulation of inventory value, stock, lead time, risk

**Risk Analysis (chart_05 ~ chart_08):**
- Regression analysis with OLS fit, 95% confidence intervals, Pearson/Spearman statistics
- Category risk profiling — Stock status distribution, safety stock coverage, risk density
- Outlier detection using Z-score (|z| >= 3) and IQR methods
- Risk quadrant — Demand intensity vs. stock coverage with urgency mapping
- Key variable pair plot colored by Stock Status

---

### Phase 5: Advanced Supply Chain Optimization

> `chart_09_to_14_advanced_supply_chain.py.txt` — Operations research techniques

**EOQ (Economic Order Quantity) — chart_09:**

```
EOQ = √(2DS / H)
```

Where: D = Annual Demand, S = Ordering Cost ($50/order), H = Holding Cost (25% of Unit Cost)

Includes: EOQ distribution by category, EOQ vs. demand curve, cost breakdown, cost curve visualization, reorder frequency, and savings potential vs. monthly ordering.

**Vendor Radar — chart_10:**
- Multi-dimensional vendor comparison (cost, coverage, OOS rate, lead time, inventory value, SKU count)
- Normalized scoring with inverted metrics (lower cost = better)
- Overall ranking table

**Monte Carlo Simulation — chart_12:**
- 5,000 simulated demand scenarios per category
- Stockout probability estimation during lead time
- Safety stock adequacy assessment with KDE overlay

**Reorder & Demand Analysis — chart_13, chart_14:**
- Reorder gap waterfall analysis (actual stock vs. reorder point)
- Replenishment priority matrix (urgency vs. unit cost, bubble = annual demand)
- Demand variability (Coefficient of Variation by category)
- Safety stock coverage funnel and service level curve
- Lead time risk heatmap (Category x Lead Time bucket)

---

### Phase 6: AI & Machine Learning

> `chart_15_to_22_ai_algorithms_analysis.py.txt` — 30 algorithms applied to supply chain data

The project catalogs 30 AI algorithms for supply chain applications. **20 are fully implemented**; 10 are documented as applicable with different data types (time-series, image, simulation environment).

#### Supervised Learning — Classification (chart_15)

Predicting `Stock_Status` (Out of Stock / Low Stock / Normal Stock):

| # | Algorithm | Category |
|---|-----------|----------|
| #2 | Logistic Regression | Linear Model |
| #3 | Decision Tree | Tree-based |
| #4 | Random Forest | Ensemble |
| #5 | SVM (Support Vector Machine) | Kernel Method |
| #6 | k-Nearest Neighbors (k-NN) | Instance-based |
| #7 | Naive Bayes | Probabilistic |
| #8 | Gradient Boosting | Ensemble |
| #9 | AdaBoost | Ensemble |
| #10 | XGBoost | Ensemble (optional) |
| #21 | ANN / MLP (Neural Network) | Deep Learning |

#### Feature Importance Analysis (chart_16)

- Random Forest, Gradient Boosting, Decision Tree feature importance
- Logistic Regression coefficient magnitude (averaged across classes)

#### Supervised Learning — Regression (chart_17)

Predicting `Inventory_Value`:

| # | Algorithm | Category |
|---|-----------|----------|
| #1 | Linear Regression | Linear Model |
| #4 | Random Forest Regressor | Ensemble |
| #8 | Gradient Boosting Regressor | Ensemble |

Includes: Actual vs. Predicted scatter, R² and RMSE comparison, residual distribution analysis.

#### Unsupervised Learning — Clustering (chart_18)

| # | Algorithm | Purpose |
|---|-----------|---------|
| #11 | K-Means | Inventory segmentation |
| #12 | Hierarchical Clustering | Agglomerative grouping |
| #13 | DBSCAN | Density-based clustering with noise detection |
| #26 | K-Means++ | Improved centroid initialization |

Includes: Elbow method + silhouette analysis, cluster vs. actual Stock Status comparison, cluster feature profiles.

#### Dimensionality Reduction (chart_19)

| # | Algorithm | Purpose |
|---|-----------|---------|
| #14 | PCA | Variance analysis, 2D projection, loading plot |
| #15 | t-SNE | Non-linear 2D embedding (n=5,000 sample) |

#### Anomaly Detection (chart_20)

| # | Algorithm | Purpose |
|---|-----------|---------|
| #27 | Autoencoder (MLP-based) | Reconstruction error anomaly detection |
| #28 | Isolation Forest | Tree-based anomaly scoring |

Includes: Cross-method agreement analysis, feature comparison (anomaly vs. normal ratio).

#### Optimization (chart_21)

| # | Algorithm | Purpose |
|---|-----------|---------|
| #30 | Genetic Algorithm | Safety stock multiplier optimization per category |

GA parameters: Population=100, Generations=80, Mutation=15%, Crossover=80%.
Includes: Convergence curve, optimal multipliers, cost savings, gene evolution tracking.

#### Algorithm Overview (chart_22)

Complete 30-algorithm suitability matrix with status (Applied / N/A), category, supply chain use case, and library reference. Includes notes on why 10 algorithms (RL #16-20, CNN #22, RNN/LSTM/Transformer #23-25, MDP #29) require different data types.

---

## Visualization Gallery

The pipeline generates **28 publication-ready charts** across all analysis phases (charts 0-22 from statistical/ML analysis + charts 23-28 from RL optimization):

### Inventory Health Dashboard
| Chart | File | Description |
|-------|------|-------------|
| `chart_0` | `chart_0_Inventory Health & Stockout Risk.png` | KPI cards, inventory efficiency scatter, ABC Pareto, stockout alert table |

### Statistical Analysis (chart_01 ~ chart_08)
| Chart | File | Description |
|-------|------|-------------|
| `chart_01` | `chart_01_correlation_matrix.png` | Pearson vs. Spearman correlation heatmaps with p-value significance |
| `chart_02` | `chart_02_distribution_analysis.png` | Histogram + KDE + Q-Q plots for 6 key metrics with Shapiro-Wilk tests |
| `chart_03` | `chart_03_vendor_performance.png` | Vendor performance box plots with ANOVA & Kruskal-Wallis tests |
| `chart_04` | `chart_04_category_vendor_heatmap.png` | Category x Vendor cross-tabulation heatmaps with Chi-square test |
| `chart_05` | `chart_05_regression_analysis.png` | OLS regression with 95% CI for 4 key variable pairs |
| `chart_06` | `chart_06_category_risk_profile.png` | Stock status, safety stock coverage, risk score distribution by category |
| `chart_07` | `chart_07_outlier_risk_analysis.png` | Z-score & IQR outlier detection, risk quadrant (demand intensity vs. coverage) |
| `chart_08` | `chart_08_pairplot_regression.png` | Key variable pair plot colored by Stock Status |

### Advanced Supply Chain Optimization (chart_09 ~ chart_14)
| Chart | File | Description |
|-------|------|-------------|
| `chart_09` | `chart_09_eoq_analysis.png` | EOQ distribution, cost curve, reorder frequency, savings potential |
| `chart_10` | `chart_10_vendor_radar.png` | Multi-dimensional vendor radar charts with overall ranking |
| `chart_11` | `chart_11_inventory_treemap.png` | Inventory value treemap + nested donut (category x vendor) |
| `chart_12` | `chart_12_monte_carlo_stockout.png` | Monte Carlo stockout probability simulation (5,000 runs per category) |
| `chart_13` | `chart_13_reorder_gap_waterfall.png` | Reorder gap violin, top 25 urgent SKUs, replenishment priority matrix |
| `chart_14` | `chart_14_demand_safety_stock.png` | Demand variability (CV), safety stock funnel, service level curve, lead time risk |

### Machine Learning (chart_15 ~ chart_22)
| Chart | File | Description |
|-------|------|-------------|
| `chart_15` | `chart_15_classification_comparison.png` | 10-algorithm accuracy ranking, CV scores, overfitting check, top-3 confusion matrices |
| `chart_16` | `chart_16_feature_importance.png` | Feature importance from Random Forest, Gradient Boosting, Decision Tree, Logistic Regression |
| `chart_17` | `chart_17_regression_prediction.png` | Actual vs. predicted scatter (3 models), R²/RMSE comparison, residual distribution |
| `chart_18` | `chart_18_clustering_analysis.png` | Elbow + silhouette, K-Means++/Hierarchical/DBSCAN results, cluster profiles |
| `chart_19` | `chart_19_pca_tsne.png` | PCA variance explained, PCA 2D (by status & category), t-SNE 2D, loading plot |
| `chart_20` | `chart_20_anomaly_detection.png` | Isolation Forest + Autoencoder anomaly detection, score distributions, cross-method agreement |
| `chart_21` | `chart_21_genetic_algorithm.png` | GA convergence, optimal safety stock multipliers, cost savings, gene evolution |
| `chart_22` | `chart_22_algorithm_overview.png` | 30-algorithm suitability matrix with status and library reference |

### Reinforcement Learning (chart_23 ~ chart_28)
| Chart | File | Description |
|-------|------|-------------|
| `chart_23` | `chart_23_rl_reward_curves.png` | Reward learning curves for all 6 agents |
| `chart_24` | `chart_24_rl_service_levels.png` | Service level improvement over training episodes |
| `chart_25` | `chart_25_rl_comparison.png` | Multi-agent performance comparison |
| `chart_26` | `chart_26_rl_convergence.png` | Convergence analysis across agents |
| `chart_27` | `chart_27_rl_reward_distribution.png` | Final reward distribution by agent |
| `chart_28` | `chart_28_rl_summary.png` | Summary comparison table of all agents |

---

## Business Insights & Findings

### Key Discoveries

1. **Inventory Imbalance** — ABC analysis reveals that ~20% of products account for ~80% of total inventory value, confirming Pareto distribution in supply chain holdings.

2. **Stockout Risk Concentration** — Monte Carlo simulations identify specific category-vendor combinations with >30% stockout probability, enabling targeted safety stock adjustments.

3. **Vendor Performance Variance** — Statistical testing (ANOVA) reveals significant cost differences across vendors for equivalent categories, suggesting procurement optimization opportunities.

4. **Predictive Power** — Ensemble methods (Random Forest, Gradient Boosting) achieve the highest accuracy in Stock Status prediction, with `Reorder_Point` and `Current_Stock` as the dominant features.

5. **Clustering-Based Segmentation** — K-Means identifies 3-4 natural inventory segments that align with but refine the rule-based ABC classification.

6. **Genetic Algorithm Optimization** — GA-optimized safety stock multipliers per category achieve measurable cost savings compared to uniform 1.0x baseline, balancing holding cost against stockout risk.

### Actionable Recommendations

| Finding | Recommendation | Expected Impact |
|---------|---------------|----------------|
| High stockout probability items | Increase safety stock by GA-optimized multipliers | Reduce stockouts by ~40% |
| Vendor cost variance | Negotiate or consolidate vendors | 5-15% cost reduction |
| Low Stock items near reorder point | Implement automated reorder triggers | Prevent revenue loss |
| Anomalous inventory patterns | Investigate for data entry errors or theft | Improve data integrity |
| EOQ vs. monthly ordering gap | Adopt EOQ-based order quantities | Reduce total inventory cost |

---

## Roadmap & Future Work

### Planned Enhancements

- [ ] **Deep Learning Integration** — Implement LSTM/GRU networks via TensorFlow/Keras for time-series demand forecasting
- [x] **Reinforcement Learning** — 6 RL agents (Q-Learning, SARSA, DQN, PPO, A2C, GA-RL Hybrid) for dynamic reorder policy optimization
- [x] **Interactive Dashboard** — Vanilla HTML/JS dashboard with i18n (EN/JA/ZH), CSV-driven KPIs, and chart gallery (`dashboard.html`)
- [x] **React SPA + Real-Time** — React 18 + TypeScript frontend with WebSocket live pipeline progress, 7 pages, trilingual i18n
- [x] **Database Backend** — SQLite via SQLAlchemy (InventorySnapshot, AnalysisResult, PipelineRun models)
- [x] **API Service** — FastAPI REST API (11 endpoints) + WebSocket for real-time updates
- [x] **Automated Alerting** — Watchdog file monitoring auto-triggers pipeline on CSV drop; stockout alerts on dashboard
- [ ] **Multi-Warehouse Support** — Extend data model to support multi-location inventory optimization
- [ ] **Supplier Lead Time Prediction** — ML model to predict actual lead times based on historical patterns
- [ ] **Docker Containerization** — Reproducible deployment with Docker Compose
- [ ] **PostgreSQL Migration** — Upgrade from SQLite to PostgreSQL for production scale

### Open Problems

| Problem | Description | Difficulty |
|---------|-------------|------------|
| Demand Seasonality | Current model assumes stationary demand; real supply chains have seasonal patterns | Medium |
| Multi-Echelon Optimization | Optimizing across multiple tiers of the supply chain simultaneously | Hard |
| Supplier Reliability Modeling | Incorporating probabilistic lead time variability into EOQ calculations | Medium |
| Real-Time Data Integration | Streaming inventory updates from ERP/WMS systems | Hard |
| Causal Inference | Moving beyond correlation to understand causal drivers of stockouts | Hard |
| RL Environment Design | Building a realistic simulation environment for reinforcement learning agents | Hard |

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.8+, TypeScript |
| **Backend** | FastAPI, uvicorn, SQLAlchemy, SQLite |
| **Frontend** | React 18, Vite 6, Tailwind CSS, Recharts, Zustand |
| **Real-Time** | WebSocket (auto-reconnect) |
| **i18n** | Custom React hook (EN / 中文 / 日本語) |
| **File Monitoring** | watchdog (auto-trigger pipeline on CSV drop) |
| **Data Processing** | pandas, numpy |
| **Statistical Analysis** | scipy (chi2, ANOVA, Kruskal-Wallis, Shapiro-Wilk, correlation tests) |
| **Machine Learning** | scikit-learn (classification, regression, clustering, anomaly detection) |
| **Reinforcement Learning** | Gymnasium, PyTorch (Q-Learning, SARSA, DQN, PPO, A2C, GA-RL) |
| **Visualization** | matplotlib, seaborn, squarify (treemap), Recharts (interactive) |
| **Optional** | xgboost (if installed, used for XGBoost classifier) |

### Python Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
squarify>=0.4.3
fastapi>=0.100.0
uvicorn>=0.22.0
sqlalchemy>=2.0.0
websockets>=11.0
watchdog>=3.0.0
gymnasium>=0.28.0
torch>=2.0.0
```

### Frontend Dependencies

```
react, react-dom, react-router-dom
recharts, zustand
tailwindcss, autoprefixer, postcss
vite, typescript
```

---

## Language Support

### React SPA — Trilingual i18n (EN / 中文 / 日本語)

The React frontend supports **3 languages** with a one-click switcher in the header. All UI labels, page titles, form labels, chart titles, and status messages are fully translated. Language preference is persisted in `localStorage`.

| Component | Translated Elements |
|-----------|-------------------|
| Sidebar | Navigation labels, subtitle |
| Header | Subtitle, dark mode toggle |
| Dashboard | KPI titles, chart titles, table headers |
| Upload | Instructions, status messages |
| Statistics | Tab labels, chart titles |
| Supply Chain | EOQ labels, Monte Carlo labels |
| ML / RL | Section titles, table headers |
| History | Column headers, status labels |

### Batch Scripts — Bilingual (Chinese + English)

Every analysis script is provided in both **Chinese** and **English** with identical functionality:

| Module | Chinese Version | English Version |
|--------|----------------|-----------------|
| ETL Pipeline | `clean_data .py` | `clean_data_en.py.txt` |
| Inventory Health | `chart_0_Inventory Health & Stockout Risk.py.txt` | `chart_0_Inventory Health & Stockout Risk.en.py.txt` |
| Statistical Analysis | `chart_01_to_08_statistical_analysis.py.txt` | `chart_01_to_08_statistical_analysis_en.py.txt` |
| Advanced Optimization | `chart_09_to_14_advanced_supply_chain.py.txt` | `chart_09_to_14_advanced_supply_chain_en.py.txt` |
| AI/ML Algorithms | `chart_15_to_22_ai_algorithms_analysis.py.txt` | `chart_15_to_22_ai_algorithms_analysis_en.py.txt` |

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add new analysis module'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

### Areas Where Help Is Needed

- Time-series demand forecasting with deep learning (LSTM/Transformer)
- PostgreSQL migration for production deployment
- Docker containerization
- Additional data quality issue generators
- Unit test coverage (Python + React)
- Documentation translations (Korean, Spanish)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by real-world supply chain data challenges in inventory management
- Statistical methods reference: *Supply Chain Management: Strategy, Planning, and Operation* (Chopra & Meindl)
- ML pipeline patterns adapted from scikit-learn best practices documentation
- EOQ model based on the classic Harris-Wilson formula (1913)

---

<div align="center">

**Built with data, driven by insight.**

If this project helped you, please consider giving it a star!

</div>
