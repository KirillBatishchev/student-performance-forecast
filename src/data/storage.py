import os
from pathlib import Path
import json
import torch
import pandas as pd
from io import BytesIO
from minio import Minio
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / "secrets" / ".env"
if env_path.exists():
    load_dotenv(env_path)

client = Minio(
    endpoint=os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ROOT_USER"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
    secure=False,
)
BUCKET = os.getenv("MINIO_BUCKET")

# Списки
def list_files(prefix: str = "") -> list:
    """Список всех файлов в бакете по префиксу"""
    objects = client.list_objects(BUCKET, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]

# Возвращает индексы всех файлов пользователей
def get_all_users() -> list:
    """Все user_id из raw/users_logs"""
    files = list_files("raw/users_logs")
    return [
        f.split("/")[-1].replace(".csv", "")
        for f in files
        if f.endswith(".csv")
    ]

# Возвращает индексы файлов, которые не использовались в обучении ранее
def unseen_users(trained_users_file: set) -> list:
    all_users = set(get_all_users())
    if trained_users_file is None:
        return list(all_users)
    
    unseen = list(all_users - trained_users_file)
    return unseen


# Чтение
def load_csv(path: str) -> pd.DataFrame:
    """Прочитать CSV из MinIO → DataFrame"""
    response = client.get_object(BUCKET, path)
    return pd.read_csv(BytesIO(response.data))


def load_json(path: str) -> dict:
    """Прочитать JSON из MinIO → dict"""
    response = client.get_object(BUCKET, path)
    return json.load(BytesIO(response.data))


def load_text(path: str) -> str:
    """Загрузить текст из MinIO"""
    response = client.get_object(BUCKET, path)
    return response.read().decode("utf-8")


# Загрузка весов и параметров для первого обучения модели
def initial_model():
    """Загрузка данных для инициализации модели"""
    weights_path = "model_weights/initial_weights.pth"
    params_path = "model_weights/initial_parameters.json"
    response_weights = client.get_object(BUCKET, weights_path)
    response_params = client.get_object(BUCKET, params_path)
    weights = torch.load(BytesIO(response_weights.data), map_location="cpu")
    params = json.load(BytesIO(response_params.data))
    return {"weights": weights, "parameters": params}


# Сохранение
def save_csv(df: pd.DataFrame, path: str):
    """Сохранить DataFrame как CSV в MinIO"""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    client.put_object(
        BUCKET, path, data=BytesIO(csv_bytes), length=len(csv_bytes)
    )


def save_json(data: dict, path: str):
    """Сохранить dict как JSON в MinIO"""
    json_bytes = json.dumps(data, indent=2).encode("utf-8")
    client.put_object(
        BUCKET, path, data=BytesIO(json_bytes), length=len(json_bytes)
    )
    
def save_text(text: str, path: str):
    """Сохранить текст в MinIO"""
    text_bytes = text.encode("utf-8")
    client.put_object(
        BUCKET, path, data=BytesIO(text_bytes), length=len(text_bytes)
    )

# Удаление
def delete_file(path: str):
    """Удалить файл из MinIO"""
    try:
        client.remove_object(BUCKET, path)
        return True
    except Exception as e:
        print(f"Ошибка удаления {path}: {e}")
        return False
    

def get_mlflow_client():
    """Получить клиент для бакета mlflow"""
    return Minio(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ROOT_USER"),
        secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
        secure=False,
    )


MLFLOW_BUCKET = "mlflow"

def list_files_mlflow(prefix: str = ""):
    """Список файлов в бакете mlflow"""
    client = get_mlflow_client()
    objects = client.list_objects(MLFLOW_BUCKET, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects]


def delete_file_mlflow(path: str):
    """Удалить файл из бакета mlflow"""
    try:
        client = get_mlflow_client()
        client.remove_object(MLFLOW_BUCKET, path)
        return True
    except Exception as e:
        print(f"Ошибка удаления {path}: {e}")
        return False
    
