"""SQLAlchemy models for ChainInsight."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Text, JSON,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL

Base = declarative_base()


class InventorySnapshot(Base):
    """One row per product per ingestion batch."""
    __tablename__ = "inventory_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), index=True, nullable=False)
    ingested_at = Column(DateTime, index=True, default=lambda: datetime.now(timezone.utc))
    product_id = Column(String(32), index=True)
    category = Column(String(32))
    unit_cost = Column(Float)
    current_stock = Column(Float)
    daily_demand_est = Column(Float)
    safety_stock_target = Column(Float)
    vendor_name = Column(String(64))
    lead_time_days = Column(Float)
    reorder_point = Column(Float)
    stock_status = Column(String(20))
    inventory_value = Column(Float)


class AnalysisResult(Base):
    """Stores KPI / chart metadata per pipeline run."""
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), index=True, nullable=False)
    analysis_type = Column(String(32), nullable=False)  # etl, stats, scm, ml
    result_json = Column(JSON)
    chart_paths = Column(JSON)  # list of chart file paths
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PipelineRun(Base):
    """Tracks each pipeline execution."""
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), unique=True, nullable=False)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    source_file = Column(String(256))
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
    etl_stats = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)


# ---- Engine & Session factory ----

engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(engine)


SessionLocal = sessionmaker(bind=engine)


def get_db():
    """Yield a DB session (for FastAPI dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
