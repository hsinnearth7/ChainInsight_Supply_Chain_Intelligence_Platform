"""Pipeline Orchestrator — coordinates ETL -> Stats -> Supply Chain -> ML -> RL."""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import pandas as pd

from app.config import RAW_DIR, CLEAN_DIR, CHARTS_DIR
from app.pipeline.etl import ETLPipeline
from app.pipeline.stats import StatisticalAnalyzer
from app.pipeline.supply_chain import SupplyChainAnalyzer
from app.pipeline.ml_engine import MLAnalyzer
from app.rl.trainer import RLTrainer
from app.rl.evaluator import RLEvaluator
from app.db.models import (
    SessionLocal, init_db,
    PipelineRun, InventorySnapshot, AnalysisResult,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates the full analysis pipeline with progress callbacks."""

    STAGES = ["etl", "stats", "supply_chain", "ml", "rl"]

    def __init__(self, on_progress: Optional[Callable] = None):
        """
        Args:
            on_progress: Optional callback(stage, status, data) for real-time updates.
        """
        self.on_progress = on_progress or (lambda *a, **kw: None)
        init_db()

    def run(self, input_path: str, batch_id: Optional[str] = None) -> dict:
        """Execute the full pipeline.

        Args:
            input_path: Path to raw CSV file.
            batch_id: Optional batch identifier (auto-generated if None).

        Returns:
            Dict with batch_id, all results, and chart paths.
        """
        batch_id = batch_id or f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info("Pipeline started — batch_id=%s, input=%s", batch_id, input_path)

        db = SessionLocal()
        run_record = PipelineRun(
            batch_id=batch_id,
            status="running",
            source_file=str(input_path),
        )
        db.add(run_record)
        db.commit()

        all_results = {"batch_id": batch_id, "stages": {}}
        charts_dir = CHARTS_DIR / batch_id
        charts_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Stage 1: ETL
            self.on_progress("etl", "running", {})
            etl = ETLPipeline()
            clean_path = CLEAN_DIR / f"{batch_id}_clean.csv"
            df_clean = etl.run(input_path, str(clean_path))
            etl_stats = etl.get_stats()
            all_results["stages"]["etl"] = etl_stats
            self._save_snapshots(db, batch_id, df_clean)
            self._save_analysis(db, batch_id, "etl", etl_stats, [])
            self.on_progress("etl", "completed", etl_stats)

            # Stage 2: Statistical Analysis
            self.on_progress("stats", "running", {})
            stats_analyzer = StatisticalAnalyzer(output_dir=str(charts_dir))
            stats_results = stats_analyzer.run_all(df_clean)
            all_results["stages"]["stats"] = stats_results
            self._save_analysis(db, batch_id, "stats", stats_results.get("kpis", {}), stats_results.get("chart_paths", []))
            self.on_progress("stats", "completed", stats_results)

            # Stage 3: Supply Chain Optimization
            self.on_progress("supply_chain", "running", {})
            sc_analyzer = SupplyChainAnalyzer(output_dir=str(charts_dir))
            sc_results = sc_analyzer.run_all(df_clean)
            all_results["stages"]["supply_chain"] = sc_results
            self._save_analysis(db, batch_id, "supply_chain", sc_results.get("results", {}), sc_results.get("chart_paths", []))
            self.on_progress("supply_chain", "completed", sc_results)

            # Stage 4: ML Analysis
            self.on_progress("ml", "running", {})
            ml_analyzer = MLAnalyzer(output_dir=str(charts_dir))
            ml_results = ml_analyzer.run_all(df_clean)
            all_results["stages"]["ml"] = ml_results
            self._save_analysis(db, batch_id, "ml", ml_results.get("results", {}), ml_results.get("chart_paths", []))
            self.on_progress("ml", "completed", ml_results)

            # Stage 5: RL Training & Evaluation
            self.on_progress("rl", "running", {})
            rl_env_kwargs = self._build_rl_env_kwargs(df_clean)
            rl_trainer = RLTrainer(
                n_episodes=300,
                episode_length=90,
                env_kwargs=rl_env_kwargs,
            )
            rl_results_raw = rl_trainer.train_all(on_progress=self.on_progress)
            comparison_data = rl_trainer.get_comparison_data()

            rl_evaluator = RLEvaluator(comparison_data, output_dir=str(charts_dir))
            rl_chart_paths = rl_evaluator.generate_all_charts()
            rl_kpis = rl_evaluator.get_kpis()

            all_results["stages"]["rl"] = {
                "kpis": rl_kpis,
                "chart_paths": rl_chart_paths,
                "comparison": comparison_data,
            }
            rl_result_data = dict(rl_kpis) if isinstance(rl_kpis, dict) else {"kpis": rl_kpis}
            rl_result_data["comparison_data"] = self._to_serializable(comparison_data)
            self._save_analysis(db, batch_id, "rl", rl_result_data, rl_chart_paths)
            self.on_progress("rl", "completed", rl_kpis)

            # Mark pipeline complete
            run_record.status = "completed"
            run_record.completed_at = datetime.now(timezone.utc)
            run_record.etl_stats = etl_stats
            db.commit()

            all_results["status"] = "completed"
            logger.info("Pipeline completed — batch_id=%s", batch_id)

        except Exception as e:
            run_record.status = "failed"
            run_record.error_message = str(e)
            run_record.completed_at = datetime.now(timezone.utc)
            db.commit()
            all_results["status"] = "failed"
            all_results["error"] = str(e)
            logger.exception("Pipeline failed — batch_id=%s", batch_id)
            self.on_progress("error", "failed", {"error": str(e)})
        finally:
            db.close()

        return all_results

    def _save_snapshots(self, db, batch_id: str, df: pd.DataFrame):
        """Save cleaned data as inventory snapshots."""
        now = datetime.now(timezone.utc)
        snapshots = []
        for _, row in df.iterrows():
            snapshots.append(InventorySnapshot(
                batch_id=batch_id,
                ingested_at=now,
                product_id=row.get("Product_ID"),
                category=row.get("Category"),
                unit_cost=row.get("Unit_Cost"),
                current_stock=row.get("Current_Stock"),
                daily_demand_est=row.get("Daily_Demand_Est"),
                safety_stock_target=row.get("Safety_Stock_Target"),
                vendor_name=row.get("Vendor_Name"),
                lead_time_days=row.get("Lead_Time_Days"),
                reorder_point=row.get("Reorder_Point"),
                stock_status=row.get("Stock_Status"),
                inventory_value=row.get("Inventory_Value"),
            ))
        db.bulk_save_objects(snapshots)
        db.commit()
        logger.info("Saved %d inventory snapshots", len(snapshots))

    def _save_analysis(self, db, batch_id: str, analysis_type: str, result_data: dict, chart_paths: list):
        """Save analysis results to DB."""
        # Convert numpy types to Python native for JSON serialization
        clean_data = self._to_serializable(result_data)
        db.add(AnalysisResult(
            batch_id=batch_id,
            analysis_type=analysis_type,
            result_json=clean_data,
            chart_paths=chart_paths,
        ))
        db.commit()

    @staticmethod
    def _build_rl_env_kwargs(df: pd.DataFrame) -> dict:
        """Build InventoryEnv kwargs from cleaned data statistics."""
        return {
            "unit_cost": float(df["Unit_Cost"].median()),
            "daily_demand_mean": float(df["Daily_Demand_Est"].mean()),
            "daily_demand_std": float(df["Daily_Demand_Est"].std()),
            "lead_time": int(df["Lead_Time_Days"].median()),
            "safety_stock": float(df["Safety_Stock_Target"].median()),
            "max_stock": float(df["Current_Stock"].quantile(0.95) * 2),
        }

    def _to_serializable(self, obj):
        """Recursively convert numpy types to Python native types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
