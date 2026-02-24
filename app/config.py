"""ChainInsight configuration."""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CHARTS_DIR = DATA_DIR / "charts"

# Ensure directories exist
for d in [RAW_DIR, CLEAN_DIR, CHARTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'chaininsight.db'}")

# Supply chain parameters
ORDERING_COST = 50       # $ per order
HOLDING_RATE = 0.25      # 25% of unit cost per year
MONTE_CARLO_SIMS = 5000
GA_POPULATION = 100
GA_GENERATIONS = 80
GA_MUTATION_RATE = 0.15
GA_CROSSOVER_RATE = 0.80

# Chart settings
CHART_DPI = 150
CHART_BG_COLOR = "#F0F2F6"
CHART_TEXT_COLOR = "#1B2A4A"
COLOR_PALETTE = {
    "primary": "#2E86C1",
    "success": "#27AE60",
    "danger": "#E74C3C",
    "warning": "#F39C12",
    "purple": "#8E44AD",
    "teal": "#1ABC9C",
    "gray": "#95A5A6",
}
STATUS_COLORS = {
    "Normal Stock": "#27AE60",
    "Low Stock": "#F39C12",
    "Out of Stock": "#E74C3C",
}
