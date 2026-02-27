"""Shared test fixtures for ChainInsight."""

import io
import os
import tempfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Set test API key before importing app
os.environ["API_KEY"] = "test-api-key"
os.environ["DATABASE_URL"] = "sqlite:///test_chaininsight.db"

from app.db.models import Base, engine
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    """Create test database tables."""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def api_key():
    """Valid API key for tests."""
    return "test-api-key"


@pytest.fixture
def auth_headers(api_key):
    """Headers with valid API key."""
    return {"X-API-Key": api_key}


@pytest.fixture
def sample_csv_content():
    """Minimal valid CSV data matching the expected schema."""
    return (
        "Product_ID,Category,Unit_Cost_Raw,Current_Stock_Raw,"
        "Daily_Demand_Est,Safety_Stock_Target,Vendor_Name,Lead_Time_Days\n"
        "P001,Electronics,$50.00,100,10,30,Tokyo Electronics,5\n"
        "P002,Home,$25.00,0,8,20,Kyoto Crafts,7\n"
        "P003,Food,$10.00,50,15,40,Hokkaido Foods,3\n"
        "P004,Office,$30.00,-5,12,25,Osaka Supplies,10\n"
        "P005,Electronics,$75.00,200,5,50,Tokyo Electronics,4\n"
    )


@pytest.fixture
def sample_csv_file(sample_csv_content):
    """Write sample CSV to a temp file and return path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(sample_csv_content)
        return f.name


@pytest.fixture
def sample_df(sample_csv_content):
    """Cleaned DataFrame from sample CSV (post-ETL)."""
    from app.pipeline.etl import ETLPipeline
    df = pd.read_csv(io.StringIO(sample_csv_content))
    etl = ETLPipeline()
    return etl.run_from_dataframe(df)
