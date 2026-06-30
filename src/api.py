from pathlib import Path
from models.predict import predict, predict_random_users
from models.train import train, finetune
from data.drift_detector import check_drift
import mlflow
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
import sys
import os
import uuid
import time
import logging
import data.storage as st
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

sys.path.insert(0, os.path.dirname(__file__))

# Метрики prometheus
drift_status = Gauge(
    "drift_status",
    "Статус дрифта: 0=stable, 1=warning, 2=drift"
)
drift_psi = Gauge(
    "drift_psi",
    "Текущее значение PSI"
)
request_count = Counter(
    "predictions_total",
    "Всего предсказаний",
    ["status"]
)
request_latency = Histogram(
    "prediction_latency_seconds",
    "Время инференса"
)
prediction_value = Histogram(
    "prediction_value",
    "Распределение предсказаний",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Загрузка секретов
env_path = Path(__file__).resolve().parent.parent / "secrets" / ".env"
load_dotenv(env_path)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://130.49.153.56:9000"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD", "")
os.environ["MLFLOW_S3_IGNORE_TLS"] = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

# PYDANTIC МОДЕЛИ


class PredictRequest(BaseModel):
    user_ids: List[str] = Field(None, min_length=1, description="Список ID пользователей")
    random_count: int = Field(None, ge=1, le=100, description="Количество случайных пользователей")


class PredictResponse(BaseModel):
    status: str
    request_id: str
    predictions: list
    timestamp: str


class FinetuneResponse(BaseModel):
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
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/predictions/recent")
async def recent_predictions(limit: int = 20):
    try:
        latest = st.load_json("logs/predictions/latest.json")
        return {"predictions": latest.get("predictions", [])[:limit]}
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        return {"predictions": []}
    
    
@app.get("/predictions/history")
async def get_predictions_history(limit: int = 50):
    """Получить предсказания из всех файлов logs/predictions/"""
    try:
        # Получаем все файлы в logs/predictions/
        files = st.list_files("logs/predictions/")
        
        # Фильтруем только файлы с предсказаниями (не latest.json и не папки)
        prediction_files = [
            f for f in files
            if f.endswith(".json")
            and "latest" not in f
            and "raw" not in f
        ]
        
        # Сортируем по времени (новые сверху)
        prediction_files = sorted(prediction_files, reverse=True)[:limit]
        
        all_predictions = []
        for file_path in prediction_files:
            try:
                data = st.load_json(file_path)
                predictions = data.get("predictions", [])
                timestamp = data.get("timestamp", "")
                model_version = data.get("model_version", "unknown")
                
                for pred in predictions:
                    pred["timestamp"] = timestamp
                    pred["model_version"] = model_version
                    pred["source_file"] = file_path
                    all_predictions.append(pred)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Ограничиваем общее количество
        all_predictions = all_predictions[:limit]
        
        return {"predictions": all_predictions, "total": len(all_predictions)}
        
    except Exception as e:
        logger.error(f"Error loading predictions history: {e}")
        return {"predictions": [], "total": 0}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """Предсказание для списка пользователей или случайных"""
    request_id = str(uuid.uuid4())[:8]
    
    # Определяем режим
    if request.random_count:
        logger.info(f"[{request_id}] Predict random: {request.random_count} users")
        results = predict_random_users(request.random_count)
    elif request.user_ids:
        logger.info(f"[{request_id}] Predict: {request.user_ids}")
        results = predict(request.user_ids)
    else:
        raise HTTPException(status_code=400, detail="Укажите user_ids или random_count")
    
    start = time.time()
    try:
        latency = time.time() - start
        request_latency.observe(latency)
        request_count.labels(status="success").inc(len(results))
        for r in results:
            if "prediction" in r:
                prediction_value.observe(r["prediction"])

        return PredictResponse(
            status="success",
            request_id=request_id,
            predictions=results,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        request_count.labels(status="error").inc()
        logger.error(f"[{request_id}] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/finetune", response_model=FinetuneResponse)
async def finetune_endpoint(background_tasks: BackgroundTasks):
    """Запуск дообучения"""
    job_id = str(uuid.uuid4())[:8]
    logger.info(f"[{job_id}] Finetune started")

    def run():
        try:
            result = finetune()
            logger.info(f"[{job_id}] Finetune completed: {result}")
        except Exception as e:
            logger.error(f"[{job_id}] Finetune failed: {e}")

    background_tasks.add_task(run)

    return FinetuneResponse(
        status="started",
        job_id=job_id,
        message="Finetune started"
    )

@app.post("/check_drift")
async def check_drift_endpoint(window_size: int = 100):
    """Запуск проверки дрифта с выбором размера окна"""
    try:
        # Проверяем допустимые значения
        valid_sizes = [10, 100, 500, 1000]
        if window_size not in valid_sizes:
            window_size = 100
        
        report = check_drift(window_size=window_size)
        
        if report:
            # Обновляем метрики Prometheus
            status_map = {"stable": 0, "warning": 1, "drift": 2}
            drift_status.set(status_map.get(report.get("overall_status", "stable"), 0))
            drift_psi.set(report.get("max_psi", 0))
            return report
        else:
            return {"status": "no_data", "message": "Недостаточно данных для проверки"}
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/drift/status")
async def get_drift_status():
    """Получить последний статус дрифта"""
    try:
        report = st.load_json("logs/drift/latest_report.json")
        return {
            "status": report.get("overall_status", "unknown"),
            "timestamp": report.get("timestamp", "unknown"),
            "max_psi": report.get("data_drift", {}).get("max_psi", 0),
            "concept_drift": report.get("concept_drift", {}).get("status", "no_data")
        }
    except Exception as e:
        return {"status": "no_data", "message": "Отчёт не найден"}


@app.get("/drift/report/text")
async def get_drift_report_text():
    """Получить текстовый отчёт о дрифте"""
    try:
        text = st.load_text("logs/drift/latest_text_report.txt")
        return Response(content=text, media_type="text/plain")
    except Exception as e:
        print(f"Error loading text report: {e}")
        return {"status": "no_data", "message": "Отчёт не найден"}


@app.get("/experiments")
async def get_experiments():
    """Список экспериментов из MLflow"""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiments = client.search_experiments()
        return {
            "experiments": [
                {
                    "name": exp.name,
                    "id": exp.experiment_id,
                    "status": exp.lifecycle_stage,
                    "runs": len(client.search_runs(experiment_ids=[str(exp.experiment_id)]))
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        logger.error(f"Error loading experiments: {e}")
        return {"experiments": []}
    
    
@app.get("/model/metrics")
async def get_model_metrics():
    """Метрики по версиям моделей из MLflow"""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Получаем все версии модели SimpleDKT
        versions = client.search_model_versions("name='SimpleDKT'")
        
        result = []
        for v in versions:
            try:
                run = client.get_run(v.run_id)
                metrics = run.data.metrics
                
                # Пробуем разные названия метрик
                accuracy = metrics.get("val_acc")
                if accuracy is None:
                    accuracy = metrics.get("accuracy", 0)
                
                loss = metrics.get("val_loss")
                if loss is None:
                    loss = metrics.get("loss", 0)
                
                # Преобразуем timestamp в строку
                timestamp = run.info.start_time
                if timestamp:
                    from datetime import datetime
                    timestamp_str = datetime.fromtimestamp(timestamp / 1000).isoformat()
                else:
                    timestamp_str = ""
                
                result.append({
                    "version": str(v.version),  # ← явно в строку
                    "accuracy": float(accuracy) if accuracy else 0,
                    "loss": float(loss) if loss else 0,
                    "timestamp": timestamp_str
                })
            except Exception as e:
                print(f"Error processing version {v.version}: {e}")
                continue
        
        # Сортируем по версии
        result = sorted(result, key=lambda x: int(x["version"]))
        
        return {"versions": result}
        
    except Exception as e:
        logger.error(f"Error loading model metrics: {e}")
        return {"versions": []}
    

@app.get("/model/version")
async def get_model_version():
    """Получить текущую версию модели"""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = client.get_latest_versions("SimpleDKT")
        if versions:
            return {
                "version": versions[0].version,
                "run_id": versions[0].run_id,
                "status": versions[0].status
            }
        return {"version": "unknown"}
    except Exception as e:
        logger.error(f"Error loading model version: {e}")
        return {"version": "error"}
    

@app.post("/train")
async def train_endpoint():
    """Запуск обучения с нуля"""
    logger.info("Training started")
    try:
        result = train()
        return {"status": "success", "result": str(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/stats")
async def get_predictions_stats():
    """Статистика по предсказаниям из всех файлов"""
    try:
        # Получаем все файлы в logs/predictions/
        files = st.list_files("logs/predictions/")
        
        # Фильтруем только файлы с предсказаниями (не latest.json и не папки)
        prediction_files = [
            f for f in files
            if f.endswith(".json")
            and "latest" not in f
            and "raw" not in f
        ]
        
        all_predictions = []
        for file_path in prediction_files:
            try:
                data = st.load_json(file_path)
                predictions = data.get("predictions", [])
                all_predictions.extend(predictions)
            except Exception as e:
                print(f"Error: {e}")
                continue

        total = len(all_predictions)
        success_count = sum(1 for p in all_predictions if p.get("will_succeed", False))
        failure_count = total - success_count
        avg_confidence = sum(p.get("prediction", 0) for p in all_predictions) / total if total > 0 else 0
        
        return {
            "total": total,
            "success_count": success_count,
            "failure_count": failure_count,
            "avg_confidence": avg_confidence
        }
    except Exception as e:
        logger.error(f"Error loading predictions stats: {e}")
        return {"total": 0, "success_count": 0, "failure_count": 0, "avg_confidence": 0}


@app.get("/data/info")
async def get_data_info():
    """Информация о данных"""
    try:
        # Всего пользователей
        all_users = st.get_all_users()
        total_users = len(all_users)
        
        # Новые пользователи (не использованные в обучении)
        try:
            history = st.load_json("logs/training/history.json")
            trained_users = set()
            for entry in history.get("history", []):
                trained_users.update(entry.get("train_users", []))
                trained_users.update(entry.get("valid_users", []))
            new_users = len(set(all_users) - trained_users)
        except Exception as e:
            print(f"Error: {e}")
            new_users = 0
        
        # Всего предсказаний
        try:
            latest = st.load_json("logs/predictions/latest.json")
            total_predictions = len(latest.get("predictions", []))
        except Exception as e:
            print(f"Error: {e}")
            total_predictions = 0
        
        return {
            "total_users": total_users,
            "new_users": new_users,
            "total_predictions": total_predictions
        }
    except Exception as e:
        logger.error(f"Error loading data info: {e}")
        return {"total_users": 0, "new_users": 0, "total_predictions": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
