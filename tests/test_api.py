import pytest
from unittest.mock import patch, MagicMock


def test_placeholder():
    """Заглушка — заменим на реальные тесты когда будет FastAPI."""
    assert True


def test_storage_unseen_users():
    """Проверяем что unseen_users возвращает список."""
    with patch("minio.Minio", return_value=MagicMock()):
        from src.data import storage
        import importlib
        importlib.reload(storage)

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