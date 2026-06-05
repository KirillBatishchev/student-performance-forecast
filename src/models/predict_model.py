import mlflow
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
import data.storage as st
import models.model as Model
import features.build_features as bs


def prepare_for_predict(raw_df, correct_dict, params, means, stds):
    """
    raw_df: один пользователь
    correct_dict: словарь ответов
    params: конфиг модели
    means, stds: сохранённые из обучения
    """
    # 1. Предобработка
    processed_df = bs.Transformation_data(raw_df, correct_dict)
    
    # 2. Нормализация сохранёнными means/stds
    processed_df[params["feature_cols"]] = (
        processed_df[params["feature_cols"]].values - means
    ) / stds
    
    # 3. Датасет
    dataset = Model.DKTSequenceDataset(
        sequences=[processed_df],
        feature_cols=params["feature_cols"],
        target_col=params["target_col"],
        max_len=params["max_len"],
    )
    
    return dataset[0]  # возвращаем один элемент


def load_model():
    return mlflow.pytorch.load_model("models:/SimpleDKT/latest")

def predict(model, raw_df, correct_dict, params):
    """
    raw_df: сырой DataFrame одного пользователя
    correct_dict: словарь правильных ответов
    params: конфиг с feature_cols, target_col, max_len
    """
    # Предобработка
    processed_df = bs.Transformation_data(raw_df, correct_dict)
    
    # Нормализация (те же means/stds, что при обучении)
    # Либо norm_params передаётся отдельно, либо используется
    # нормализация из saved_config
    
    return 0

def predict_batch(model, user_ids, params):
    """Предсказание для списка пользователей"""
    results = {}
    for user_id in user_ids:
        raw_df = st.load_csv(f"raw/users_logs/{user_id}.csv")
        questions_df = st.load_csv("raw/contents/questions.csv")
        correct_dict = dict(zip(questions_df['question_id'], questions_df['correct_answer']))
        results[user_id] = predict(model, raw_df, correct_dict, params)
    return results