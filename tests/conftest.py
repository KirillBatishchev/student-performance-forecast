import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://test:test@localhost:5432/test_mlops"
)
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MLFLOW_MODEL_URI", "models:/SimpleDKT/1")
os.environ.setdefault("MODEL_VERSION", "test")