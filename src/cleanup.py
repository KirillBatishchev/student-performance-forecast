from dotenv import load_dotenv
import data.storage as st
from mlflow.tracking import MlflowClient
import mlflow
import sys
import os
import argparse
from pathlib import Path
import shutil
import time

sys.path.insert(0, os.path.dirname(__file__))


env_path = Path(__file__).resolve().parent.parent.parent / "secrets" / ".env"
if env_path.exists():
    load_dotenv(env_path)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")
# os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://130.49.153.56:9000"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER", "")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD", "")
os.environ["MLFLOW_S3_IGNORE_TLS"] = os.getenv("MLFLOW_S3_IGNORE_TLS", "true")
os.environ["MLFLOW_S3_VERIFY_SSL"] = "false"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))


def clean_registered_models():
    """Удалить ВСЕ зарегистрированные модели из MLflow Registry"""
    print("\nУдаление моделей из MLflow Registry...")

    try:
        client = MlflowClient()
        models = client.search_registered_models()

        if not models:
            print("Нет моделей для удаления")
            return

        for model in models:
            print(f"Удаление модели: {model.name}")
            # Получаем все версии модели
            versions = client.search_model_versions(f"name='{model.name}'")
            for version in versions:
                try:
                    client.delete_model_version(model.name, version.version)
                    print(f"  Удалена версия {version.version}")
                except Exception as e:
                    print(f"  Ошибка удаления версии {version.version}: {e}")
            try:
                client.delete_registered_model(model.name)
                print(f"  Удалена модель: {model.name}")
            except Exception as e:
                print(f"  Ошибка удаления модели {model.name}: {e}")

        print("Модели удалены")

    except Exception as e:
        print(f"Ошибка: {e}")


def clean_mlflow_artifacts():
    """Удалить ВСЕ артефакты из бакета mlflow"""
    print("\nОчистка артефактов из бакета mlflow...")

    try:
        files = st.list_files_mlflow("")

        if not files:
            print("Нет артефактов")
            return

        deleted_count = 0
        for file_path in files:
            try:
                st.delete_file_mlflow(file_path)
                print(f"Удалён: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"Ошибка удаления {file_path}: {e}")

        print(f"Удалено {deleted_count} артефактов")

    except Exception as e:
        print(f"Ошибка: {e}")


def clean_experiment_runs(experiment_name="StudentPerformance"):
    """Удалить все runs внутри эксперимента"""
    print(f"\nОчистка runs в эксперименте '{experiment_name}'...")

    try:
        client = MlflowClient()

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Эксперимент '{experiment_name}' не найден")
            return

        exp_id = experiment.experiment_id
        runs = client.search_runs(experiment_ids=[str(exp_id)])

        if not runs:
            print(f"Нет runs в эксперименте '{experiment_name}'")
            return

        for run in runs:
            try:
                client.delete_run(run.info.run_id)
                print(f"Удалён run: {run.info.run_id[:8]}")
            except Exception as e:
                print(f"Ошибка удаления run {run.info.run_id[:8]}: {e}")

        print(f"Удалено {len(runs)} runs")

    except Exception as e:
        print(f"Ошибка: {e}")


def clean_models_in_experiment(experiment_name="StudentPerformance"):
    """
    ПОЛНАЯ ОЧИСТКА МОДЕЛЕЙ В ЭКСПЕРИМЕНТЕ
    """
    # 1. Удаляем модели из Registry
    clean_registered_models()

    # 2. Удаляем артефакты из бакета mlflow
    clean_mlflow_artifacts()

    # 3. Удаляем runs внутри эксперимента
    clean_experiment_runs(experiment_name)


def reset_logs():
    """Удалить все логи из MinIO, кроме logs/drift/reference.json"""
    print("\nОчистка логов")

    try:
        all_files = st.list_files("logs/")
    except Exception as e:
        print(f"Нет файлов в logs/: {e}")
        return

    if not all_files:
        print("Папка logs/ пуста")
        return

    deleted_count = 0

    for file_path in all_files:
        if file_path == "logs/drift/reference.json":
            print(f"Сохранён: {file_path}")
            continue

        try:
            st.delete_file(file_path)
            print(f"Удалён: {file_path}")
            deleted_count += 1
        except Exception as e:
            print(f"Ошибка удаления {file_path}: {e}")

    print(f"Удалено {deleted_count} файлов")


def clean_all():
    """Полная очистка (модели + логи)"""
    print("\n" + "=" * 50)
    print("ПОЛНАЯ ОЧИСТКА")
    print("=" * 50)

    clean_registered_models()
    clean_mlflow_artifacts()
    clean_experiment_runs()
    reset_logs()


def main():
    parser = argparse.ArgumentParser(description="Очистка системы")
    parser.add_argument("--all", action="store_true", help="Очистить всё (модели + логи)")
    parser.add_argument("--models", action="store_true", help="Очистить модели (Registry + runs + артефакты)")
    parser.add_argument("--experiment", action="store_true", help="Очистить модели в эксперименте")
    parser.add_argument("--logs", action="store_true", help="Очистить логи")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        print("\nУкажите: --all, --models, --experiment, --logs")
        return

    confirm = input("\nВНИМАНИЕ: Это удалит данные!\nПродолжить? (yes/no): ")
    if confirm.lower() not in ["yes", "y"]:
        print("Отмена")
        return

    if args.all:
        clean_all()
    else:
        if args.experiment:
            clean_models_in_experiment()
        if args.models:
            clean_registered_models()
            clean_mlflow_artifacts()
            clean_experiment_runs()
        if args.logs:
            reset_logs()

    print("\nГотово!")


if __name__ == "__main__":
    main()
