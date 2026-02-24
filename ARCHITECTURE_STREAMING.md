# ChainInsight 動態即時串流儀表板 — 架構升級方案

## 現狀分析

目前的 ChainInsight 是一個 **批次處理 (Batch Processing)** 架構：

```
CSV 輸入 → ETL 清洗 → 統計分析 → 供應鏈優化 → ML 分析 → 23 張 PNG → 靜態 HTML 儀表板
```

每次都要手動跑 5 個 Python 腳本，圖表是靜態 PNG，儀表板只能讀已存在的圖和 CSV。

---

## 目標架構：Event-Driven Streaming Pipeline

```
                    ┌─────────────────────────────────────────────────┐
                    │            ChainInsight Live Platform            │
                    └─────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌───────────────────────┐
  │  Data Source  │────▸│  Ingestion   │────▸│   Event Bus / Queue   │
  │              │     │   Gateway    │     │  (Redis Stream /      │
  │ • CSV Upload │     │              │     │   Celery / RQ)        │
  │ • API Push   │     │ • Validate   │     │                       │
  │ • ERP/WMS    │     │ • Timestamp  │     │ • on_data_arrived     │
  │ • Scheduled  │     │ • Queue      │     │ • on_etl_complete     │
  │   Polling    │     │              │     │ • on_analysis_done    │
  └──────────────┘     └──────────────┘     └──────────┬────────────┘
                                                       │
                    ┌──────────────────────────────────┐│
                    │        Pipeline Orchestrator      ││
                    │        (自動觸發流程)              │◂┘
                    └──────┬───────┬───────┬───────┬───┘
                           │       │       │       │
                    ┌──────▾──┐ ┌──▾─────┐ ┌▾─────┐ ┌▾──────────┐
                    │  ETL    │ │ Stats  │ │ SCM  │ │  ML + RL  │
                    │ Pipeline│ │Analysis│ │Optim.│ │  Engine   │
                    │         │ │        │ │      │ │           │
                    │ 8-step  │ │chart   │ │EOQ   │ │ 30 algos │
                    │ cleaning│ │01-08   │ │Monte │ │ + RL new  │
                    │         │ │Carlo │ │           │
                    └────┬────┘ └───┬────┘ └──┬───┘ └─────┬─────┘
                         │         │          │           │
                         ▾         ▾          ▾           ▾
                    ┌─────────────────────────────────────────┐
                    │          Result Store (DB/Cache)         │
                    │  • SQLite / PostgreSQL (結構化資料)       │
                    │  • Redis (即時 KPI 快取)                 │
                    │  • File Store (圖表 PNG/SVG)             │
                    └──────────────────┬──────────────────────┘
                                       │
                    ┌──────────────────▾──────────────────────┐
                    │       Real-Time Dashboard (Web)          │
                    │                                          │
                    │  ┌──────────┐  ┌──────────┐  ┌────────┐│
                    │  │WebSocket │  │ Plotly/  │  │ SSE    ││
                    │  │ Server   │  │ Echarts  │  │ Push   ││
                    │  └──────────┘  └──────────┘  └────────┘│
                    │                                          │
                    │  • KPI 即時更新 (無需刷新)                │
                    │  • 互動式圖表 (zoom/filter/drill-down)   │
                    │  • Pipeline 進度條                       │
                    │  • RL Agent 決策即時視覺化                │
                    └──────────────────────────────────────────┘
```

---

## 核心升級分為 5 大模組

---

### 模組 1：Web 後端 + Pipeline Orchestrator

**技術選型：FastAPI + Celery (或 RQ)**

```python
# app/main.py — FastAPI 入口
from fastapi import FastAPI, UploadFile, WebSocket
from app.pipeline import trigger_full_pipeline
from app.ws_manager import ConnectionManager

app = FastAPI(title="ChainInsight Live")
ws_manager = ConnectionManager()

@app.post("/api/ingest")
async def ingest_data(file: UploadFile):
    """新資料上傳後自動觸發完整 pipeline"""
    raw_path = save_upload(file)
    # 非同步觸發：ETL → Stats → SCM → ML → RL
    task = trigger_full_pipeline.delay(raw_path)
    return {"task_id": task.id, "status": "pipeline_started"}

@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    """WebSocket 即時推送分析結果到前端"""
    await ws_manager.connect(websocket)
    # 每個 pipeline 階段完成時推送更新
```

**Pipeline Orchestrator（自動化流程）：**

```python
# app/pipeline.py — 事件驅動的 Pipeline
from celery import chain

def trigger_full_pipeline(raw_path: str):
    """資料進來 → 自動跑完全部流程"""
    workflow = chain(
        etl_task.s(raw_path),           # Step 1: 8-step ETL
        stats_task.s(),                  # Step 2: 統計分析
        supply_chain_task.s(),           # Step 3: 供應鏈優化
        ml_task.s(),                     # Step 4: ML 30 algos
        rl_task.s(),                     # Step 5: RL 動態決策 ← 新增!
        publish_results.s(),             # Step 6: 推送到 Dashboard
    )
    return workflow.apply_async()
```

---

### 模組 2：即時互動式儀表板

**技術選型比較：**

| 方案 | 技術 | 優點 | 缺點 |
|------|------|------|------|
| **A (推薦)** | FastAPI + React/Vue + Plotly.js | 完全自定義、最佳 UX、WebSocket 原生 | 開發量較大 |
| **B (快速)** | Streamlit / Dash | Python 全棧、開發快 | 自定義受限、並行能力弱 |

**推薦方案 A 的前端架構：**

```
前端 (React + TypeScript)
├── components/
│   ├── KPIDashboard.tsx        # 6 個 KPI 即時卡片
│   ├── PipelineProgress.tsx    # 流程進度條 (ETL → Stats → ML → RL)
│   ├── InteractiveCharts/
│   │   ├── CorrelationMatrix.tsx   # Plotly heatmap (可互動)
│   │   ├── MonteCarloSim.tsx       # 動態 Monte Carlo 模擬
│   │   ├── EOQOptimizer.tsx        # 拖動參數即時重算 EOQ
│   │   ├── RLDecisionViz.tsx       # RL Agent 決策軌跡
│   │   └── InventoryHealthMap.tsx  # ABC 分類 + 風險熱力圖
│   ├── DataUploader.tsx         # 拖拽上傳 CSV
│   └── AlertPanel.tsx           # 即時缺貨警報
├── hooks/
│   └── useWebSocket.ts          # WebSocket 連線管理
└── stores/
    └── dashboardStore.ts        # 全域狀態管理
```

**關鍵互動功能：**
- 圖表全部從靜態 PNG → **Plotly.js / ECharts 互動式**（zoom、filter、drill-down）
- WebSocket 即時推送：pipeline 每階段完成 → 自動更新對應圖表
- 參數可調：使用者可以在前端調整 EOQ 的 S/H 參數、Monte Carlo 模擬次數、GA 的代數等

---

### 模組 3：Reinforcement Learning 引擎（核心新增）

目前有 GA（遺傳演算法）做靜態安全庫存最佳化。RL 的價值在於：**它能根據環境回饋持續學習最佳策略，不只是一次性最佳化。**

#### RL 應用場景設計

```
┌─────────────────────────────────────────────────────────┐
│                  RL for Supply Chain                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  場景 1: 動態補貨決策 (Dynamic Reorder Agent)             │
│  ─────────────────────────────────────────                │
│  State:  [庫存量, 需求趨勢, Lead_Time, 季節, 缺貨天數]    │
│  Action: [不補貨, 補少量, 補 EOQ 量, 緊急大量補貨]        │
│  Reward: -持有成本 - 缺貨損失 + 服務水平獎勵             │
│                                                          │
│  場景 2: 動態定價/折扣策略 (Pricing Agent)                │
│  ─────────────────────────────────────────                │
│  State:  [庫存量, 需求彈性, 競爭價格, 剩餘保質期]         │
│  Action: [維持原價, 小幅降價, 大幅促銷, 加價]             │
│  Reward: 毛利 × 銷量 - 庫存持有成本                      │
│                                                          │
│  場景 3: 多倉庫調撥 (Inventory Balancing Agent)           │
│  ─────────────────────────────────────────                │
│  State:  [各倉庫存量, 各倉需求, 調撥成本矩陣]            │
│  Action: [不調撥, A→B 調撥, B→A 調撥, 調撥量]           │
│  Reward: -總缺貨成本 - 調撥成本 + 服務水平獎勵           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### RL 實作架構

```python
# app/rl/environment.py — 供應鏈模擬環境 (Gymnasium 介面)
import gymnasium as gym
import numpy as np

class InventoryEnv(gym.Env):
    """
    模擬一個品類的庫存管理環境
    - 每個 time step = 1 天
    - Agent 決定補貨量
    - 環境回傳需求（隨機）、缺貨懲罰、持有成本
    """
    def __init__(self, category_data, config):
        self.demand_mean = category_data['daily_demand_mean']
        self.demand_std = category_data['daily_demand_std']
        self.unit_cost = category_data['avg_unit_cost']
        self.lead_time = category_data['avg_lead_time']
        self.holding_rate = config.get('holding_rate', 0.25 / 365)
        self.stockout_penalty = config.get('stockout_penalty', 2.0)

        # State: [current_stock, pending_orders, days_since_last_order,
        #         demand_trend_7d, stockout_flag]
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(5,), dtype=np.float32
        )
        # Action: 離散補貨決策
        # 0=不補, 1=補0.5×EOQ, 2=補1×EOQ, 3=補1.5×EOQ, 4=補2×EOQ
        self.action_space = gym.spaces.Discrete(5)

    def step(self, action):
        order_qty = action * 0.5 * self.eoq
        demand = max(0, np.random.normal(self.demand_mean, self.demand_std))
        self.stock = max(0, self.stock - demand)
        self._process_arrivals()
        if action > 0:
            self._place_order(order_qty)
        holding_cost = self.stock * self.unit_cost * self.holding_rate
        stockout_cost = max(0, demand - self.stock) * self.unit_cost * self.stockout_penalty
        reward = -(holding_cost + stockout_cost)
        return self._get_obs(), reward, False, False, self._get_info()
```

#### RL 演算法實作路線

```
Level 1 (基礎):
├── Q-Learning (Tabular)      — 離散化 state/action，建立 Q-table
├── SARSA                      — On-policy 版本的 Q-Learning
└── ── 用既有的 7 個品類各訓練一個 agent

Level 2 (進階):
├── DQN (Deep Q-Network)       — Neural Network 取代 Q-table
├── Double DQN                 — 解決 overestimation 問題
└── Dueling DQN                — 分離 state-value 和 advantage

Level 3 (前沿):
├── PPO (Proximal Policy Optimization)  — 最穩定的 Policy Gradient
├── A2C / A3C (Actor-Critic)            — 並行訓練
└── SAC (Soft Actor-Critic)             — 適合連續動作空間
```

#### RL vs GA 的互補關係

```
┌──────────────────────────────────────────────────────┐
│              GA + RL 混合策略                          │
├──────────────────────────────────────────────────────┤
│                                                      │
│  GA (現有): 靜態最佳化                                │
│  ├── 輸入: 歷史統計數據                              │
│  ├── 輸出: 最佳安全庫存乘數 [0.5x ~ 3.0x]            │
│  ├── 優點: 全域搜索、不受梯度限制                    │
│  └── 限制: 不能適應環境變化、一次性計算              │
│                                                      │
│  RL (新增): 動態決策                                  │
│  ├── 輸入: 即時庫存狀態 (每天/每小時)                │
│  ├── 輸出: 當下最佳行動 (補貨?多少?何時?)           │
│  ├── 優點: 持續學習、適應季節/趨勢變化              │
│  └── 限制: 需要模擬環境、訓練時間較長               │
│                                                      │
│  ★ 混合方案:                                         │
│  GA 找出全域最優參數 → 作為 RL 的初始策略           │
│  RL 在 GA 基礎上做即時微調 → 適應當前環境           │
│  每月用新資料重跑 GA → 更新 RL 的策略基準           │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

### 模組 4：資料持久化 + 時間序列

**從單一 CSV → 資料庫 + 版本化**

```python
# app/db/models.py
from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class InventorySnapshot(Base):
    """每次資料匯入都保存一份快照，支援時間序列分析"""
    __tablename__ = 'inventory_snapshots'
    id = Column(Integer, primary_key=True)
    ingested_at = Column(DateTime, index=True)
    batch_id = Column(String, index=True)
    product_id = Column(String, index=True)
    category = Column(String)
    unit_cost = Column(Float)
    current_stock = Column(Float)
    daily_demand_est = Column(Float)
    safety_stock_target = Column(Float)
    vendor_name = Column(String)
    lead_time_days = Column(Float)
    reorder_point = Column(Float)
    stock_status = Column(String)
    inventory_value = Column(Float)

class AnalysisResult(Base):
    """儲存每次分析結果，支援歷史比較"""
    __tablename__ = 'analysis_results'
    id = Column(Integer, primary_key=True)
    batch_id = Column(String, index=True)
    analysis_type = Column(String)
    result_json = Column(JSON)
    chart_path = Column(String)
    created_at = Column(DateTime)

class RLEpisode(Base):
    """記錄 RL Agent 的訓練與決策歷程"""
    __tablename__ = 'rl_episodes'
    id = Column(Integer, primary_key=True)
    agent_type = Column(String)
    category = Column(String)
    episode = Column(Integer)
    total_reward = Column(Float)
    avg_stockout_rate = Column(Float)
    avg_holding_cost = Column(Float)
    policy_snapshot = Column(JSON)
    trained_at = Column(DateTime)
```

有了時間序列資料，就能解鎖之前標為 N/A 的演算法：

| 演算法 | 現狀 | 升級後 |
|--------|------|--------|
| RNN (#23) | N/A (缺時間序列) | **可用** — 多批次資料形成時間序列 |
| LSTM (#24) | N/A | **可用** — 需求預測 |
| Transformer (#25) | N/A | **可用** — 多品類注意力機制 |
| Q-Learning (#16) | N/A | **可用** — RL 環境已建立 |
| SARSA (#17) | N/A | **可用** |
| DQN (#18) | N/A | **可用** |
| Policy Gradient (#19) | N/A | **可用** |
| Actor-Critic (#20) | N/A | **可用** |
| MDP (#29) | N/A | **可用** — InventoryEnv 就是 MDP |

**升級後：30 個演算法全部 Applied！**

---

### 模組 5：自動化觸發 + 監控

```python
# app/triggers.py — 多種資料輸入觸發方式

# 1. 檔案監控 (Watch 資料夾)
from watchdog.observers import Observer
class CSVWatcher:
    """監控指定資料夾，新 CSV 出現時自動觸發 pipeline"""
    def on_created(self, event):
        if event.src_path.endswith('.csv'):
            trigger_full_pipeline.delay(event.src_path)

# 2. 排程觸發 (每日/每小時)
from celery.schedules import crontab
CELERYBEAT_SCHEDULE = {
    'daily-inventory-sync': {
        'task': 'app.pipeline.sync_from_erp',
        'schedule': crontab(hour=6, minute=0),
    },
    'hourly-rl-update': {
        'task': 'app.pipeline.rl_incremental_train',
        'schedule': crontab(minute=0),
    },
}

# 3. API Webhook (ERP/WMS 主動推送)
@app.post("/webhook/inventory-update")
async def erp_webhook(payload: InventoryUpdate):
    trigger_full_pipeline.delay(payload.data)

# 4. 手動上傳 (Dashboard UI)
# → 見模組 1 的 /api/ingest 端點
```

---

## 技術棧總覽

| 層級 | 技術 | 用途 |
|------|------|------|
| **Web Framework** | FastAPI | REST API + WebSocket |
| **Task Queue** | Celery + Redis | 非同步 pipeline 執行 |
| **Database** | SQLite (dev) → PostgreSQL (prod) | 結構化資料 + 時間序列 |
| **Cache** | Redis | 即時 KPI + Session |
| **RL Framework** | Gymnasium + Stable-Baselines3 | RL 環境 + 預訓練演算法 |
| **Deep Learning** | PyTorch | DQN / PPO / LSTM |
| **前端** | React + Plotly.js (或 Streamlit 快速版) | 互動式儀表板 |
| **即時通訊** | WebSocket (FastAPI 內建) | 推送更新到前端 |
| **檔案監控** | Watchdog | 自動偵測新 CSV |
| **排程** | Celery Beat | 定時觸發 |
| **容器化** | Docker + docker-compose | 一鍵部署 |

---

## 實作優先順序

### Phase 1 (MVP — 最小可行產品)

```
① 將現有分析腳本模組化 (function-based → class-based)
② FastAPI 後端 + CSV 上傳 API
③ Pipeline orchestrator (ETL → Stats → ML 自動串接)
④ Streamlit 快速版即時儀表板 (取代靜態 HTML)
⑤ SQLite 儲存歷史分析結果
```

### Phase 2 (RL 引擎)

```
⑥ InventoryEnv (Gymnasium 環境)
⑦ Q-Learning + DQN 實作
⑧ PPO / A2C 進階 RL
⑨ GA→RL 混合策略 (GA 初始化 + RL 微調)
⑩ RL 決策視覺化 (reward curve, policy heatmap)
```

### Phase 3 (即時串流)

```
⑪ WebSocket 即時推送
⑫ React 互動式前端 (取代 Streamlit)
⑬ Watchdog 檔案監控自動觸發
⑭ Celery Beat 排程
⑮ 即時警報系統 (缺貨、異常)
```

### Phase 4 (生產級)

```
⑯ PostgreSQL 遷移
⑰ Docker + docker-compose
⑱ LSTM/Transformer 需求預測 (需累積時間序列)
⑲ Multi-warehouse RL agent
⑳ CI/CD + 自動化測試
```

---

## 預期新目錄結構

```
ChainInsight/
├── app/
│   ├── main.py                  # FastAPI 入口
│   ├── config.py                # 統一設定
│   ├── pipeline/
│   │   ├── orchestrator.py      # Pipeline 排程器
│   │   ├── etl.py               # 重構的 ETL (from clean_data.py)
│   │   ├── stats.py             # 重構的統計分析
│   │   ├── supply_chain.py      # 重構的供應鏈優化
│   │   └── ml_engine.py         # 重構的 ML 分析
│   ├── rl/
│   │   ├── environment.py       # Gymnasium InventoryEnv
│   │   ├── agents/
│   │   │   ├── q_learning.py    # Tabular Q-Learning
│   │   │   ├── dqn.py           # Deep Q-Network
│   │   │   ├── ppo.py           # Proximal Policy Optimization
│   │   │   └── hybrid_ga_rl.py  # GA + RL 混合
│   │   ├── trainer.py           # 訓練管理器
│   │   └── evaluator.py         # 策略評估 + 比較
│   ├── db/
│   │   ├── models.py            # SQLAlchemy 模型
│   │   ├── session.py           # DB 連線管理
│   │   └── migrations/          # Alembic 遷移
│   ├── api/
│   │   ├── routes.py            # REST API 路由
│   │   └── websocket.py         # WebSocket 管理
│   └── services/
│       ├── chart_service.py     # 圖表產生 (Plotly)
│       ├── alert_service.py     # 警報系統
│       └── export_service.py    # PDF/Excel 匯出
├── frontend/                    # React + TypeScript (Phase 3)
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   └── stores/
│   └── package.json
├── data/
│   ├── raw/                     # 上傳的原始 CSV
│   ├── clean/                   # ETL 清洗後
│   └── charts/                  # 產生的圖表
├── tests/
├── docker-compose.yml
├── requirements.txt             # 更新的依賴
└── README.md
```

---

## 升級前後對比

| 面向 | 現在 | 升級後 |
|------|------|--------|
| **觸發方式** | 手動跑 5 個腳本 | 上傳 CSV / API / 排程自動觸發 |
| **圖表** | 23 張靜態 PNG | 互動式 Plotly (zoom/filter/drill-down) |
| **儀表板** | 靜態 HTML 讀 CSV | WebSocket 即時串流更新 |
| **最佳化** | GA 一次性計算 | GA 全域搜索 + RL 持續動態調整 |
| **ML 覆蓋** | 20/30 演算法 | **30/30 全部 Applied** |
| **資料儲存** | 單一 CSV | DB + 時間序列 + 歷史版本 |
| **部署** | `python xxx.py` | Docker 一鍵啟動 |
