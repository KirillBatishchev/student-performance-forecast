import mlflow
from pathlib import Path
import os
from dotenv import load_dotenv
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
import data.storage as st
import models.model as Model
import features.build_features as bs

env_path = Path(__file__).resolve().parent.parent.parent / "secrets" / ".env"
if env_path.exists():
    load_dotenv(env_path)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://130.49.153.56:9000"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("MINIO_ROOT_USER")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("MINIO_ROOT_PASSWORD")
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
os.environ["MLFLOW_S3_VERIFY_SSL"] = "false"

def prepare_data(user_ids, params, correct_dict):
    sequences = []

    for user_id in user_ids:
        raw_df = st.load_csv(f"raw/users_logs/{user_id}.csv")
        processed_df = bs.Transformation_data(raw_df, correct_dict)
        sequences.append(processed_df)
        
    means = params["normalization"]["means"]
    stds = params["normalization"]["stds"]
    feature_cols = params["data"]["feature_cols"]
    target_col = params["data"]["target_col"]
    max_len=params["data"]["max_len"]

    for seq in sequences:
        seq[feature_cols] = (
            seq[feature_cols].values - means
        ) / stds

    split_idx = int(len(sequences) * 0.8)

    train_ids = user_ids[:split_idx]
    valid_ids = user_ids[split_idx:]

    train_seq = sequences[:split_idx]
    valid_seq = sequences[split_idx:]

    train_dataset = Model.DKTSequenceDataset(
        sequences=train_seq,
        feature_cols=feature_cols,
        target_col=target_col,
        max_len=max_len,
    )

    valid_dataset = Model.DKTSequenceDataset(
        sequences=valid_seq,
        feature_cols=feature_cols,
        target_col=target_col,
        max_len=max_len,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    return train_dataloader, valid_dataloader, train_ids, valid_ids


def train_core(model, params, train_loader, valid_loader):
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["training"]["lr"])
    scheduler = CyclicLR(
        optimizer,
        base_lr=params["training"]["lr"],
        max_lr=params["training"]["max_lr"],
        step_size_up=params["training"]["lr_step"],
    )
    criterion = nn.BCELoss()

    eps = 1e-5
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    mlflow.log_params(
        {
            "lr": params["training"]["lr"],
            "lr_max": params["training"]["max_lr"],
            "lr_step": params["training"]["lr_step"],
            "epochs": params["training"]["epochs"],
            "batch_size": params["training"].get("batch_size", 32),
            "optimizer": params["training"]["optimizer"],
            "scheduler": params["training"]["scheduler"],
        }
    )

    for epoch in range(params["training"]["epochs"]):
        # Train
        model.train()
        train_loss = 0
        train_acc = 0
        train_count = 0

        for batch in train_loader:
            X = batch["X"].to("cpu")
            y = batch["y"].to("cpu")
            mask = batch["mask"].to("cpu")

            y_binary = (y >= 0.5).float()

            pred = model(X, mask)
            pred_next = pred[:, :-1]
            y_next = y_binary[:, 1:]
            mask_next = mask[:, :-1]

            loss_mask = mask_next.bool()
            if loss_mask.sum() > 0:
                loss = criterion(pred_next[loss_mask], y_next[loss_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

                pred_class = (pred_next > 0.5).float()
                train_acc += (
                    (pred_class[loss_mask] == y_next[loss_mask]).sum().item()
                )
                train_count += loss_mask.sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / train_count if train_count > 0 else 0

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        val_precision = 0
        val_recall = 0
        val_count = 0

        with torch.no_grad():
            for batch in valid_loader:
                X = batch["X"].to("cpu")
                y = batch["y"].to("cpu")
                mask = batch["mask"].to("cpu")

                y_binary = (y >= 0.5).float()

                pred = model(X, mask)
                pred_next = pred[:, :-1]
                y_next = y_binary[:, 1:]
                mask_next = mask[:, :-1]

                loss_mask = mask_next.bool()
                if loss_mask.sum() > 0:
                    loss = criterion(pred_next[loss_mask], y_next[loss_mask])
                    val_loss += loss.item()

                    pred_class = (pred_next > 0.5).float()
                    val_acc += (
                        (pred_class[loss_mask] == y_next[loss_mask])
                        .sum()
                        .item()
                    )
                    tp = (
                        (
                            (pred_class[loss_mask] == 1)
                            & (y_next[loss_mask] == 1)
                        )
                        .sum()
                        .item()
                    )
                    fp = (
                        (
                            (pred_class[loss_mask] == 1)
                            & (y_next[loss_mask] == 0)
                        )
                        .sum()
                        .item()
                    )
                    fn = (
                        (
                            (pred_class[loss_mask] == 0)
                            & (y_next[loss_mask] == 1)
                        )
                        .sum()
                        .item()
                    )

                    val_precision += tp / (tp + fp) if (tp + fp) > 0 else 0
                    val_recall += tp / (tp + fn) if (tp + fn) > 0 else 0
                    val_count += loss_mask.sum().item()

                    avg_val_loss = val_loss / len(valid_loader)
                    avg_val_acc = val_acc / val_count if val_count > 0 else 0
                    avg_val_precision = val_precision / len(valid_loader)
                    avg_val_recall = val_recall / len(valid_loader)

        mlflow.log_metrics(
            {
                "train_loss": avg_train_loss,
                "train_acc": avg_train_acc,
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc,
                "val_precision": avg_val_precision,
                "val_recall": avg_val_recall,
            },
            step=epoch,
        )

        if avg_val_loss - eps < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch

    model.load_state_dict(best_model_state)

    mlflow.log_metrics(
        {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
        }
    )

    return model


def freeze_for_finetune(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.output.parameters():
        param.requires_grad = True

    for param in model.transformer.layers[-1].parameters():
        param.requires_grad = True


def run_training(mode="train", model=None):
    mlflow.set_tracking_uri("http://130.49.153.56:5001")
    questions_df = st.load_csv("raw/contents/questions.csv")
    correct_dict = dict(
        zip(questions_df["question_id"], questions_df["correct_answer"])
    )
    
    initial = st.initial_model()
    params = initial["parameters"]
    if mode == "train":
        model = Model.SimpleDKT(
            len(params["data"]["feature_cols"]),
            params["architecture"]["hidden_dim"],
            params["architecture"]["num_layers"],
            params["architecture"]["n_heads"],
            params["architecture"]["dropout"],
        )
        model.load_state_dict(initial["weights"])
    else:
        client = mlflow.tracking.MlflowClient()
        
        runs = client.search_runs(
            experiment_ids=["0"],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            raise Exception("No runs found for finetune")
        
        latest_run = runs[0]
        run_id = latest_run.info.run_id
        
        # Загружаем модель из артефактов run
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")

    all_users = st.get_all_users()
    seen_users = st.load_json("raw/trained_users.json")
    unseen = st.unseen_users(seen_users, all_users)

    if len(unseen) == 0:
        return {"status": "skipped", "message": "Нет новых пользователей"}

    sample_users = random.sample(unseen, min(50, len(unseen)))
    train_loader, valid_loader, train_ids, valid_ids = prepare_data(
        sample_users, params, correct_dict
    )

    seen_users["trained_users"] = list(
        set(seen_users["trained_users"]) | set(train_ids)
    )
    seen_users["valid_users"] = list(
        set(seen_users["valid_users"]) | set(valid_ids)
    )
    st.save_json(seen_users, "raw/trained_users.json")

    if mode == "finetune":
        freeze_for_finetune(model)

    with mlflow.start_run(run_name=mode):
        mlflow.log_params({
        "input_dim": params["architecture"]["input_dim"],
        "hidden_dim": params["architecture"]["hidden_dim"],
        "num_layers": params["architecture"]["num_layers"],
        "n_heads": params["architecture"]["n_heads"],
        "dropout": params["architecture"]["dropout"],
        
        "lr": params["training"]["lr"],
        "max_lr": params["training"]["max_lr"],
        "epochs": params["training"]["epochs"],
        "batch_size": params["training"]["batch_size"],
        "optimizer": params["training"]["optimizer"],
        
        "max_len": params["data"]["max_len"],
        })

        mlflow.log_param("feature_cols", json.dumps(params["data"]["feature_cols"]))
        mlflow.log_param("normalization_means", json.dumps(params["normalization"]["means"]))
        mlflow.log_param("normalization_stds", json.dumps(params["normalization"]["stds"]))
    
        model = train_core(model, params, train_loader, valid_loader)
        
        if mode == "train":
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="SimpleDKT"
            )
        elif mode == "finetune":
            run_id = mlflow.active_run().info.run_id
            
            versions = client.search_model_versions("name='SimpleDKT'")
            next_version = len(versions) + 1
            
            client.create_model_version(
                name="SimpleDKT",
                source=f"runs:/{run_id}/model",
                run_id=run_id,
            )
    

    return model


def train():
    return run_training(mode="train")


def finetune():
    
    return run_training(mode="finetune")


