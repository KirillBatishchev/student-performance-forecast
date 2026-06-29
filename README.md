# Student Performance Forecast

MLOps project for predicting student performance based on their interaction history with an educational platform.

## Overview

A SimpleDKT (Deep Knowledge Tracing) transformer model trained on the EdNet3 dataset. Based on a student's interaction history with tasks, the model predicts the probability of successfully completing the next block of assignments.

## Architecture

```
GitHub → CI/CD (GitHub Actions) → ghcr.io (Docker Registry)
                                          ↓
                              Argo CD (GitOps)
                                          ↓
                              Kubernetes (k3s)
                              ├── FastAPI service
                              ├── Streamlit Web UI
                              ├── Prometheus
                              └── Grafana
```

External services running on the server:

- **MLflow** — experiment tracking and model artifact storage
- **MinIO** — S3-compatible storage for data and predictions
- **PostgreSQL** — database backend for MLflow
- **Portainer** — Docker container management
- **pgAdmin 4** — PostgreSQL database management

## Service URLs

| Service | URL |
|---------|-----|
| FastAPI (Swagger) | http://130.49.153.56:8888/docs |
| Web UI (Streamlit) | http://130.49.153.56:8501 |
| MLflow | http://130.49.153.56:5001 |
| MinIO Console | http://130.49.153.56:9001 |
| Argo CD | https://130.49.153.56:9090 |
| Prometheus | http://130.49.153.56:9091 |
| Grafana | http://130.49.153.56:3000 |
| Portainer | https://130.49.153.56:9443 |
| pgAdmin 4 | http://130.49.153.56:5050 |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | SimpleDKT transformer (PyTorch) |
| Dataset | EdNet3 |
| API | FastAPI + Uvicorn |
| Web UI | Streamlit |
| Containerization | Docker |
| Orchestration | Kubernetes (k3s) |
| GitOps | Argo CD |
| Monitoring | Prometheus + Grafana |
| Experiment tracking | MLflow |
| Storage | MinIO (S3) |
| Database | PostgreSQL |
| CI/CD | GitHub Actions |

==============================

Project Organization
------------

├── LICENSE
├── Makefile                   <- Makefile with commands like `make data` or `make train`
├── README.md                  <- The top-level README for developers using this project.
├── Dockerfile                 <- Docker image for FastAPI service
├── Dockerfile.ui              <- Docker image for Streamlit Web UI
├── tox.ini                    <- tox file with settings for running tox; see tox.readthedocs.io
├── setup.py                   <- makes project pip installable (pip install -e .) so src can be imported
├── requirements.txt           <- The requirements file for reproducing the analysis environment
│
├── .dvc/                      <- DVC configuration and cache
├── .github/
│   └── workflows/
│       ├── lint.yml           <- Linter (flake8) CI check
│       ├── tests.yml          <- Tests (pytest) CI check
│       └── docker.yml         <- Docker build and push to ghcr.io
│
├── data
│   ├── external               <- Data from third party sources.
│   ├── interim                <- Intermediate data that has been transformed.
│   ├── processed              <- The final, canonical data sets for modeling.
│   └── raw                    <- The original, immutable data dump.
│
├── docs                       <- A default Sphinx project; see sphinx-doc.org for details
│
├── k8s/
│   ├── deployment.yaml        <- Kubernetes Deployment for FastAPI (auto-updated by CI)
│   ├── service.yaml           <- Kubernetes Service for FastAPI
│   ├── ui-deployment.yaml     <- Kubernetes Deployment for Streamlit UI
│   └── ui-service.yaml        <- Kubernetes Service for Streamlit UI
│
├── models/
│   └── initial_model/         <- Initial model weights and parameters
│       ├── initial_parameters.json
│       └── initial_weights.pth
│
├── notebooks/
│   ├── Baseline.ipynb         <- Baseline model experiments
│   ├── EDA.ipynb              <- Exploratory data analysis
│   └── train_model.ipynb      <- Model training experiments
│
├── secrets/                   <- Local secrets (gitignored)
│   └── .env                   <- Environment variables (MINIO, MLFLOW credentials)
│
├── src                        <- Source code for use in this project.
│   ├── __init__.py            <- Makes src a Python module
│   │
│   ├── api.py                 <- FastAPI service (/predict, /retrain, /health, /metrics)
│   │
│   ├── data                   <- Scripts to download or generate data
│   │   ├── make_dataset.py
│   │   └── storage.py         <- MinIO operations (upload, download, list predictions)
│   │
│   ├── features               <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models                 <- Scripts to train models and then use trained models to make predictions
│   │   ├── __init__.py
│   │   ├── model.py           <- SimpleDKT transformer model definition
│   │   ├── predict.py         <- Model inference
│   │   └── train.py           <- Training and fine-tuning with MLflow tracking
│   │
│   ├── ui/                    <- Streamlit Web UI
│   │   └── app.py             <- Pages: Inference, Predictions, Experiments, Drift
│   │
│   └── visualization          <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tests/
    ├── conftest.py            <- pytest fixtures and configuration
    └── test_api.py            <- API endpoint tests

--------

## Web UI

Streamlit interface available on port 8501:

- **Inference** — predict for a list of students
- **Predictions** — recent prediction history
- **Experiments** — trigger model retraining
- **Drift** — data drift monitoring 

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
