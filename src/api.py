import sys
import os
import uuid
import logging
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

import mlflow
import data.storage as st
from models.train import train, finetune
from models.predict import predict

# settings

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://130.49.153.56:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "MinioSecretPassword123"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

mlflow.set_tracking_uri("http://130.49.153.56:5001")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# PYDANTIC МОДЕЛИ

class PredictRequest(BaseModel):
    user_ids: List[str] = Field(..., min_items=1, description="Список ID пользователей")

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