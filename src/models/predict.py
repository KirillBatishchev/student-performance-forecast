import models.model as Model
import features.build_features as bs
from data.drift_detector import check_drift
import data.storage as st
import argparse
import hashlib
import mlflow
import numpy as np
import torch
import os
import sys
import random
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0, "src")

env_path = Path(__file__).resolve().parent.parent.parent / "secrets" / ".env"
load_dotenv(env_path)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD", "")
os.environ["MLFLOW_S3_IGNORE_TLS"] = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

def update_windows(record):
    """
    Обновить скользящие окна (in-place)
    """
    window_sizes = [10, 100, 500, 1000]
    
    feature_vector = record["features"]
    
    window_record = {
        "user_id": record["user_id"],
        "features": feature_vector,
        "prediction": record["prediction"],
        "timestamp": record["timestamp"]
    }
    
    for size in window_sizes:
        window_path = f"logs/windows/last_{size}.json"
        
        try:
            window = st.load_json(window_path)
        except:
            window = {"size": size, "data": [], "last_update": None}
        
        window["data"].append(window_record)
        if len(window["data"]) > size:
            window["data"] = window["data"][-size:]
        window["last_update"] = datetime.now().isoformat()
        
        try:
            st.save_json(window, window_path)
        except Exception as e:
            print(f" Ошибка обновления окна {size}: {e}")


def save_prediction_log(user_id, features, prediction, model_version):
    """
    Сохранить лог предсказания для детекции дрифта
    """   
    record = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "model_version": model_version,
        "features": features.tolist() if isinstance(features, np.ndarray) else features,
        "prediction": prediction,
        "record_id": hashlib.md5(f"{user_id}_{datetime.now().timestamp()}".encode()).hexdigest()[:8]
    }
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = f"logs/predictions/raw/{date_str}/{record['record_id']}.json"
    try:
        st.save_json(record, path)
    except Exception as e:
        print(f"Ошибка сохранения лога: {e}")

    update_windows(record)
    
    return record


def get_model_version():
    """Получить версию текущей модели из MLflow"""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = client.get_latest_versions("SimpleDKT")
        if versions:
            return versions[0].version
    except:
        pass
    return "unknown"


def load_model():
    """Загрузить последнюю версию модели из MLflow Registry"""
    try:
        model = mlflow.pytorch.load_model("models:/SimpleDKT/latest")
        model.eval()
        return model
    except Exception as e:
        print(f"Проблема с загрузкой модели: {e}")
        initial = st.initial_model()
        params = initial["parameters"]
        model = Model.SimpleDKT(
            len(params["data"]["feature_cols"]),
            params["architecture"]["hidden_dim"],
            params["architecture"]["num_layers"],
            params["architecture"]["n_heads"],
            params["architecture"]["dropout"],
        )
        model.load_state_dict(initial["weights"])
        model.eval()
        return model


def predict_random_users(count: int = 10):
    """Предсказание для определенного количества случайных пользователей"""
    all_users = st.get_all_users()
    
    if len(all_users) < count:
        count = len(all_users)
    
    random_users = random.sample(all_users, count)
    return predict(random_users)


def predict(user_ids):
    """Прогноз для списка пользователей"""
    print(f"\nПрогноз для {len(user_ids)} пользователей")
    
    # 1. Загрузка данных
    params = st.initial_model()["parameters"]
    correct_dict = bs.Transform_questions_dict(st.load_csv("raw/contents/questions.csv"))
    model = load_model()
    
    # 2. Получаем версию модели
    model_version = get_model_version()

    feature_cols = params["data"]["feature_cols"]
    means = params["normalization"]["means"]
    stds = params["normalization"]["stds"]
    max_len = params["data"]["max_len"]

    results = []

    for user_id in user_ids:
        try:
            # Загрузка и трансформация
            df = bs.Transformation_data(st.load_csv(f"raw/users_logs/{user_id}.csv"), correct_dict)
            if df.empty:
                raise ValueError("Нет данных")
            
            # Нормализация
            df[feature_cols] = (df[feature_cols] - means) / stds

            # Подготовка последовательности
            features = df[feature_cols].values[-max_len:]
            if len(features) < max_len:
                features = np.pad(features, ((max_len - len(features), 0), (0, 0)), mode='constant')

            # Прогноз
            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            mask = torch.ones(1, max_len, dtype=torch.bool)

            with torch.no_grad():
                pred = torch.sigmoid(model(X, mask))[0, -1].item()

            result = {
                "user_id": user_id,
                "prediction": round(pred, 4),
                "will_succeed": pred > 0.5
            }
            results.append(result)
            
            status = 'успех' if pred > 0.5 else 'риск'
            print(f"  {user_id}: {pred:.3f} - {status}")
            
            # --- СОХРАНЯЕМ ЛОГ ДЛЯ ДРИФТА ---
            save_prediction_log(user_id, features, pred, model_version)

        except Exception as e:
            results.append({"user_id": user_id, "error": str(e)})
            print(f" {user_id}: {e}")

    # Сохранение результатов
    output = {
        "timestamp": datetime.now().isoformat(),
        "model_version": model_version,
        "total": len(results),
        "predictions": results
    }
    
    try:
        st.save_json(output, f"logs/predictions/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        st.save_json(output, "logs/predictions/latest.json")
    except Exception as e:
        print(f"Ошибка сохранения результатов: {e}")
        
    try:
        window = st.load_json("logs/windows/last_100.json")
        if len(window.get("data", [])) >= 100:
            
            check_drift(window_size=100)
            
    except Exception as e:
        print(f"  Ошибка при проверке данных: {e}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предсказание успешности выполнения задания")
    parser.add_argument(
        "--users",
        nargs="+",
        help="Список user_id (например: u10200 u19678)"
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        help="Количество случайных пользователей (например: --random 5)"
    )
    args = parser.parse_args()

    if args.users:
        predict(args.users)
    elif args.random > 0:
        predict_random_users(args.random)
    else:
        print("Укажите --users или --random")
        parser.print_help()
