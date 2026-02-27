"""Pipeline service — encapsulates pipeline run management."""

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.config import PipelineStatus
from app.db.models import PipelineRun


def generate_batch_id() -> str:
    """Generate a unique batch ID."""
    return f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def create_pipeline_run(db: Session, batch_id: str, source_file: str) -> PipelineRun:
    """Create a new PipelineRun record in queued state."""
    run_record = PipelineRun(
        batch_id=batch_id,
        status=PipelineStatus.QUEUED.value,
        source_file=source_file,
    )
    db.add(run_record)
    db.commit()
    return run_record


def get_run_by_batch(db: Session, batch_id: str) -> PipelineRun | None:
    """Look up a pipeline run by batch ID."""
    return db.query(PipelineRun).filter(PipelineRun.batch_id == batch_id).first()
