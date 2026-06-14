# Качаем python
FROM python:3.10-slim

WORKDIR /app

# Установка "C" для библиотек pandas/numpy
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем и качаем данные из requirements и setup
COPY setup.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Безопасность
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_S3_ENDPOINT_URL=""
ENV MINIO_ROOT_USER=""
ENV MINIO_ROOT_PASSWORD=""
ENV MLFLOW_S3_IGNORE_TLS="true"
ENV PYTHONPATH=/app/src

# Запускаем от имени нового пользователя
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Запускаем сервис на порте 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]