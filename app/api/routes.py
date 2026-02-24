"""FastAPI routes — REST API for pipeline execution and results retrieval."""

import asyncio
import json
import shutil
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import RAW_DIR, CHARTS_DIR
from app.db.models import get_db, SessionLocal, PipelineRun, AnalysisResult, InventorySnapshot
from app.pipeline.orchestrator import PipelineOrchestrator
from app.ws.manager import manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["pipeline"])


# ---------- Progress bridge: sync pipeline thread -> async WS ----------

def _make_ws_progress_callback(batch_id: str, loop: asyncio.AbstractEventLoop):
    """Create a sync callback that bridges to async WS broadcast.

    The pipeline runs in a thread via asyncio.to_thread(). This callback
    uses run_coroutine_threadsafe to push messages to the WS event loop.
    """
    stage_index = {s: i for i, s in enumerate(PipelineOrchestrator.STAGES)}
    total = len(PipelineOrchestrator.STAGES)

    def callback(stage: str, status: str, data: dict):
        idx = stage_index.get(stage, 0)
        if status == "completed":
            pct = int(((idx + 1) / total) * 100)
        elif status == "running":
            pct = int((idx / total) * 100)
        else:
            pct = 0

        msg_type = "pipeline:failed" if status == "failed" else f"pipeline:{status}"
        if status in ("running", "completed"):
            msg_type = f"pipeline:{'progress' if status == 'running' else 'completed' if idx == total - 1 and status == 'completed' else 'progress'}"

        msg = manager.build_message(
            msg_type=msg_type,
            batch_id=batch_id,
            stage=stage,
            status=status,
            progress_pct=pct,
            data=_safe_serialize(data),
        )
        asyncio.run_coroutine_threadsafe(
            manager.broadcast_to_batch(batch_id, msg), loop
        )

    return callback


def _safe_serialize(obj):
    """Make data JSON-safe (strip large arrays, convert numpy)."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return {"summary": str(obj)[:500]}


# ---------- Ingest (non-blocking) ----------

@router.post("/ingest")
async def ingest_csv(file: UploadFile = File(...)):
    """Upload a CSV file and trigger the full analysis pipeline (non-blocking)."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    # Save uploaded file
    raw_path = RAW_DIR / file.filename
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info("File uploaded: %s", raw_path)

    # Create batch ID and queued pipeline run record
    batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    db = SessionLocal()
    try:
        run_record = PipelineRun(batch_id=batch_id, status="queued", source_file=str(raw_path))
        db.add(run_record)
        db.commit()
    finally:
        db.close()

    # Launch pipeline in background thread via asyncio
    loop = asyncio.get_running_loop()
    progress_cb = _make_ws_progress_callback(batch_id, loop)
    orchestrator = PipelineOrchestrator(on_progress=progress_cb)

    asyncio.create_task(
        asyncio.to_thread(orchestrator.run, str(raw_path), batch_id)
    )

    return {"batch_id": batch_id, "status": "queued", "message": "Pipeline started. Connect to WS for real-time progress."}


# ---------- Ingest from path (for watchdog) ----------

async def trigger_pipeline_from_path(file_path: str):
    """Trigger a pipeline run from a file path (called by watchdog)."""
    batch_id = f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    db = SessionLocal()
    try:
        run_record = PipelineRun(batch_id=batch_id, status="queued", source_file=file_path)
        db.add(run_record)
        db.commit()
    finally:
        db.close()

    # Broadcast watchdog detection
    await manager.broadcast_global(
        manager.build_message(
            msg_type="watchdog:detected",
            batch_id=batch_id,
            data={"file": file_path},
        )
    )

    loop = asyncio.get_running_loop()
    progress_cb = _make_ws_progress_callback(batch_id, loop)
    orchestrator = PipelineOrchestrator(on_progress=progress_cb)
    asyncio.create_task(
        asyncio.to_thread(orchestrator.run, file_path, batch_id)
    )
    return batch_id


# ---------- Pipeline Status (WS fallback) ----------

@router.get("/runs/{batch_id}/status")
def get_run_status(batch_id: str, db: Session = Depends(get_db)):
    """Quick status poll for a pipeline run (fallback when WS unavailable)."""
    run = db.query(PipelineRun).filter(PipelineRun.batch_id == batch_id).first()
    if not run:
        raise HTTPException(404, "Run not found")

    # Determine current stage from analysis results
    analyses = db.query(AnalysisResult).filter(AnalysisResult.batch_id == batch_id).all()
    completed_stages = [a.analysis_type for a in analyses]
    all_stages = PipelineOrchestrator.STAGES
    progress_pct = int((len(completed_stages) / len(all_stages)) * 100) if all_stages else 0

    return {
        "batch_id": run.batch_id,
        "status": run.status,
        "progress_pct": progress_pct,
        "completed_stages": completed_stages,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "error": run.error_message,
    }


# ---------- Analysis by type ----------

@router.get("/runs/{batch_id}/analysis/{analysis_type}")
def get_analysis(batch_id: str, analysis_type: str, db: Session = Depends(get_db)):
    """Get KPIs + chart_paths for a specific analysis stage."""
    valid_types = ["etl", "stats", "supply_chain", "ml", "rl"]
    if analysis_type not in valid_types:
        raise HTTPException(400, f"Invalid analysis_type. Must be one of: {valid_types}")

    result = db.query(AnalysisResult).filter(
        AnalysisResult.batch_id == batch_id,
        AnalysisResult.analysis_type == analysis_type,
    ).first()
    if not result:
        raise HTTPException(404, f"Analysis '{analysis_type}' not found for batch {batch_id}")

    return {
        "batch_id": batch_id,
        "analysis_type": analysis_type,
        "kpis": result.result_json,
        "chart_paths": result.chart_paths,
        "created_at": result.created_at.isoformat() if result.created_at else None,
    }


# ---------- Inventory data for interactive charts ----------

@router.get("/runs/{batch_id}/data")
def get_inventory_data(batch_id: str, db: Session = Depends(get_db)):
    """Get inventory snapshot rows for interactive charting."""
    snapshots = db.query(InventorySnapshot).filter(
        InventorySnapshot.batch_id == batch_id
    ).all()
    if not snapshots:
        raise HTTPException(404, "No data found for this batch")

    return [
        {
            "product_id": s.product_id,
            "category": s.category,
            "unit_cost": s.unit_cost,
            "current_stock": s.current_stock,
            "daily_demand_est": s.daily_demand_est,
            "safety_stock_target": s.safety_stock_target,
            "vendor_name": s.vendor_name,
            "lead_time_days": s.lead_time_days,
            "reorder_point": s.reorder_point,
            "stock_status": s.stock_status,
            "inventory_value": s.inventory_value,
        }
        for s in snapshots
    ]


# ---------- Existing endpoints (unchanged) ----------

@router.get("/runs")
def list_runs(db: Session = Depends(get_db)):
    """List all pipeline runs."""
    runs = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(50).all()
    return [
        {
            "batch_id": r.batch_id,
            "status": r.status,
            "source_file": r.source_file,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        }
        for r in runs
    ]


@router.get("/runs/{batch_id}")
def get_run(batch_id: str, db: Session = Depends(get_db)):
    """Get details of a specific pipeline run."""
    run = db.query(PipelineRun).filter(PipelineRun.batch_id == batch_id).first()
    if not run:
        raise HTTPException(404, "Run not found")

    analyses = db.query(AnalysisResult).filter(AnalysisResult.batch_id == batch_id).all()

    return {
        "batch_id": run.batch_id,
        "status": run.status,
        "source_file": run.source_file,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "etl_stats": run.etl_stats,
        "error": run.error_message,
        "analyses": [
            {
                "type": a.analysis_type,
                "results": a.result_json,
                "chart_paths": a.chart_paths,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in analyses
        ],
    }


@router.get("/runs/{batch_id}/kpis")
def get_kpis(batch_id: str, db: Session = Depends(get_db)):
    """Get KPI results for a specific run."""
    result = db.query(AnalysisResult).filter(
        AnalysisResult.batch_id == batch_id,
        AnalysisResult.analysis_type == "stats",
    ).first()
    if not result:
        raise HTTPException(404, "Stats results not found")
    return result.result_json


@router.get("/runs/{batch_id}/charts")
def list_charts(batch_id: str):
    """List all chart files for a batch."""
    charts_dir = CHARTS_DIR / batch_id
    if not charts_dir.exists():
        raise HTTPException(404, "Charts directory not found")
    charts = sorted(charts_dir.glob("*.png"))
    return [{"name": c.name, "path": str(c)} for c in charts]


@router.get("/runs/{batch_id}/charts/{chart_name}")
def get_chart(batch_id: str, chart_name: str):
    """Serve a specific chart image."""
    chart_path = CHARTS_DIR / batch_id / chart_name
    if not chart_path.exists():
        raise HTTPException(404, "Chart not found")
    return FileResponse(str(chart_path), media_type="image/png")


@router.get("/latest/kpis")
def get_latest_kpis(db: Session = Depends(get_db)):
    """Get KPIs from the most recent completed pipeline run."""
    run = db.query(PipelineRun).filter(
        PipelineRun.status == "completed"
    ).order_by(PipelineRun.completed_at.desc()).first()
    if not run:
        raise HTTPException(404, "No completed runs found")

    result = db.query(AnalysisResult).filter(
        AnalysisResult.batch_id == run.batch_id,
        AnalysisResult.analysis_type == "stats",
    ).first()
    if not result:
        raise HTTPException(404, "Stats results not found")

    return {
        "batch_id": run.batch_id,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "kpis": result.result_json,
    }


@router.get("/history/kpis")
def get_kpi_history(limit: int = 20, db: Session = Depends(get_db)):
    """Get KPI history across multiple pipeline runs for trend analysis."""
    runs = db.query(PipelineRun).filter(
        PipelineRun.status == "completed"
    ).order_by(PipelineRun.completed_at.desc()).limit(limit).all()

    history = []
    for run in runs:
        result = db.query(AnalysisResult).filter(
            AnalysisResult.batch_id == run.batch_id,
            AnalysisResult.analysis_type == "stats",
        ).first()
        if result:
            history.append({
                "batch_id": run.batch_id,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "kpis": result.result_json,
            })
    return history
