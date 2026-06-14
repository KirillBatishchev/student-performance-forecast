from pathlib import Path
from models.predict import predict
from models.train import train, finetune
import mlflow
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
import sys
import os
import uuid
import logging
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

# Загрузка секретов
env_path = Path(__file__).resolve().parent.parent / "secrets" / ".env"
load_dotenv(env_path)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD", "")
os.environ["MLFLOW_S3_IGNORE_TLS"] = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# PYDANTIC МОДЕЛИ


class PredictRequest(BaseModel):
    user_ids: List[str] = Field(..., min_items=1,
                                description="Список ID пользователей")


class PredictResponse(BaseModel):
    status: str
    request_id: str
    predictions: list
    timestamp: str


class RetrainResponse(BaseModel):
    status: str
    job_id: str
    message: str

# FASTAPI


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API started")
    yield
    logger.info("API stopped")

app = FastAPI(title="MLOps API", lifespan=lifespan)

# endpoints


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """Предсказание для списка пользователей"""
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Predict: {request.user_ids}")

    try:
        results = predict(request.user_ids)
        return PredictResponse(
            status="success",
            request_id=request_id,
            predictions=results,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain", response_model=RetrainResponse)
async def retrain_endpoint(background_tasks: BackgroundTasks):
    """Запуск дообучения"""
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"[{job_id}] Retrain started")

    def run():
        try:
            result = finetune()
            logger.info(f"[{job_id}] Retrain completed: {result}")
        except Exception as e:
            logger.error(f"[{job_id}] Retrain failed: {e}")

    background_tasks.add_task(run)

    return RetrainResponse(
        status="started",
        job_id=job_id,
        message="Retraining started"
    )


@app.post("/train")
async def train_endpoint():
    """Запуск обучения с нуля"""
    logger.info("Training started")
    try:
        result = train()
        return {"status": "success", "result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
