import os
import json
import torch
import pandas as pd
from io import BytesIO
from minio import Minio
from dotenv import load_dotenv

load_dotenv("secrets/.env")

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


def get_all_users() -> list:
    """Все user_id из raw/"""
    files = list_files("raw/")
    return [
        f.split("/")[-1].replace(".csv", "")
        for f in files
        if f.endswith(".csv")
    ]


def unseen_users(trained_users_file: dict, all_users: list):
    seen = set(
        trained_users_file["trained_users"] | trained_users_file["valid_users"]
    )
    all = set(all_users)
    unseen = list(all - seen)
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


def initial_model():
    """Загрузка данных для инициализации модели"""
    weights_path = "model_weights/initial_weights.pth"
    params_path = "model_weights/initial_model_config.json"
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
