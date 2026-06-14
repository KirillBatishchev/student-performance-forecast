import models.model as Model
import features.build_features as bs
import data.storage as st
from datetime import datetime
import argparse
import mlflow
import numpy as np
import torch
import os
import sys
sys.path.insert(0, "src")


# Настройка
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://130.49.153.56:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "MinioSecretPassword123"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

mlflow.set_tracking_uri("http://130.49.153.56:5001")


def load_model():
    """Загрузить модель"""
    try:
        model = mlflow.pytorch.load_model(
            "models:/SimpleDKT/latest", map_location="cpu")
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


def predict(user_ids):
    """Прогноз для списка пользователей"""
    # Загрузка данных
    params = st.initial_model()["parameters"]
    correct_dict = bs.Transform_questions_dict(
        st.load_csv("raw/contents/questions.csv"))
    model = load_model()

    feature_cols = params["data"]["feature_cols"]
    means = params["normalization"]["means"]
    stds = params["normalization"]["stds"]
    max_len = params["data"]["max_len"]

    results = []

    for user_id in user_ids:
        try:
            # Загрузка и трансформация
            df = bs.Transformation_data(
                st.load_csv(f"raw/users_logs/{user_id}.csv"), correct_dict)
            if df.empty:
                raise ValueError("No data")

            # Нормализация
            df[feature_cols] = (df[feature_cols] - means) / stds

            # Подготовка последовательности
            features = df[feature_cols].values[-max_len:]
            if len(features) < max_len:
                features = np.pad(
                    features,
                    ((max_len - len(features), 0), (0, 0)),
                    mode='constant'
                )

            # Прогноз
            X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            mask = torch.ones(1, max_len, dtype=torch.bool)

            with torch.no_grad():
                pred = torch.sigmoid(model(X, mask))[0, -1].item()

            results.append({
                "user_id": user_id,
                "prediction": round(pred, 4),
                "will_succeed": pred > 0.5
            })
            status = 'успех' if pred > 0.5 else 'риск'
            print(f"{user_id}: {pred:.3f} - {status}")

        except Exception as e:
            results.append({"user_id": user_id, "error": str(e)})
            print(f"{user_id}: {e}")

    # Сохранение
    output = {"timestamp": datetime.now().isoformat(), "predictions": results}
    st.save_json(
        output,
        f"predictions/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--users",
        nargs="+",
        required=True,
        help="Список user_id")
    args = parser.parse_args()

    predict(args.users)
