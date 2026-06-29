import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def client():
    with patch("minio.Minio", return_value=MagicMock()):
        with patch("mlflow.set_tracking_uri"):
            with patch("mlflow.MlflowClient", return_value=MagicMock()):
                with patch.dict("sys.modules", {
                    "models.predict": MagicMock(),
                    "models.train": MagicMock(),
                    "data.storage": MagicMock(),
                }):
                    from src.api import app
                    yield TestClient(app)


def test_health(client):
    """Проверяем что /health возвращает ok"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_correct_structure(client):
    """Проверяем структуру ответа /predict"""
    mock_result = [
        {"user_id": "u10013", "prediction": 0.721, "will_succeed": True}
    ]
    with patch("src.api.predict", return_value=mock_result):
        response = client.post(
            "/predict",
            json={"user_ids": ["u10013"]}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "request_id" in data
    assert "predictions" in data
    assert "timestamp" in data


def test_predict_empty_users(client):
    """Проверяем что пустой список user_ids возвращает ошибку"""
    response = client.post(
        "/predict",
        json={"user_ids": []}
    )
    assert response.status_code == 422


def test_predict_multiple_users(client):
    """Проверяем предсказание для нескольких пользователей"""
    mock_result = [
        {"user_id": "u10013", "prediction": 0.721, "will_succeed": True},
        {"user_id": "u1020", "prediction": 0.622, "will_succeed": True},
    ]
    with patch("src.api.predict", return_value=mock_result):
        response = client.post(
            "/predict",
            json={"user_ids": ["u10013", "u1020"]}
        )
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2


def test_retrain_returns_job_id(client):
    """Проверяем что /retrain возвращает job_id"""
    response = client.post("/retrain")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    assert "job_id" in data
    assert "message" in data


def test_storage_unseen_users():
    """Проверяем что unseen_users возвращает список."""
    with patch("minio.Minio", return_value=MagicMock()):
        from src.data import storage

        trained = {
            "trained_users": {"u1", "u2"},
            "valid_users": {"u3"},
        }
        all_users = ["u1", "u2", "u3", "u4", "u5"]

        result = storage.unseen_users(trained, all_users)

        assert isinstance(result, list)
        assert "u4" in result
        assert "u5" in result
        assert "u1" not in result