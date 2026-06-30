import sys
import os
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import mlflow
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / "secrets" / ".env"
load_dotenv(env_path)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")
#os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://130.49.153.56:9000"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD", "")
os.environ["MLFLOW_S3_IGNORE_TLS"] = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

import data.storage as st
from data.drift_detector import check_drift
from models.train import finetune

def has_new_predictions(hours=1):
    """Были ли новые предсказания за последний час"""
    try:
        window = st.load_json("logs/windows/last_100.json")
        data = window.get("data", [])
        if not data:
            return False
        
        last_time = datetime.fromisoformat(data[-1]["timestamp"])
        hours_since = (datetime.now() - last_time).total_seconds() / 3600
        
        return hours_since <= hours
    except:
        return False


def was_training_recently(hours=1):
    """Было ли обучение за последний час"""
    try:
        history = st.load_json("logs/training/history.json")
        entries = history.get("history", [])
        if not entries:
            return False
        
        last_time = datetime.fromisoformat(entries[-1]["timestamp"])
        hours_since = (datetime.now() - last_time).total_seconds() / 3600
        
        return hours_since < hours
    except:
        return False


def has_enough_new_users(min_users=5):
    """Есть ли новые пользователи для переобучения"""
    try:
        all_users = set(st.get_all_users())
        trained_users = set()
        
        history = st.load_json("logs/training/history.json")
        for entry in history.get("history", []):
            trained_users.update(entry.get("train_users", []))
            trained_users.update(entry.get("valid_users", []))
        
        new_users = all_users - trained_users
        return len(new_users) >= min_users
    except:
        return False

def job():
    """
    Проверяет условия и при дрифте запускает дообучение
    """
    print(f"\n[{datetime.now().isoformat()}] Проверка...")
    
    # 1. Проверяем условия
    if not has_new_predictions():
        print("Нет новых предсказаний")
        return
    
    if not has_enough_new_users():
        print("Недостаточно новых пользователей")
        return
    
    # 2. Проверяем дрифт
    report = check_drift()
    
    if report is None:
        print("  Нет данных о дрифте")
        return
    
    status = report.get("data_drift", {}).get("overall_status", "stable")
    
    # 3. Если есть дрифт → дообучаем
    if status == "drift":
        print("Дрифт обнаружен! Запуск дообучения...")
        try:
            result = finetune()
            print(f"  Дообучение завершено: {result}")
        except Exception as e:
            print(f"  Ошибка дообучения: {e}")
    else:
        print(f"  Дрифта нет (статус: {status})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Выполнить один раз")
    parser.add_argument("--interval", type=int, default=60, help="Интервал в минутах")
    
    args = parser.parse_args()
    
    if args.once:
        job()
    else:
        print(f"Scheduler запущен. Проверка каждые {args.interval} минут.")

        job()
        
        while True:
            time.sleep(args.interval * 60)
            job()