import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import data.storage as st

# Пороги дрифта
PSI_WARNING = 0.1   # жёлтый
PSI_CRITICAL = 0.2  # красный


def load_reference():
    """    Загрузить эталон (из reference.json)"""
    try:
        return st.load_json("logs/drift/reference.json")
    except Exception as e:
        print(f"Error: {e}")
        return None


def load_window(size=100):
    """Загрузить скользящее окно"""
    try:
        window = st.load_json(f"logs/windows/last_{size}.json")
        return window.get("data", [])
    except Exception as e:
        print(f"Error: {e}")
        return []


def calculate_psi(reference_vals, current_vals, bins=10):
    """
    Расчёт PSI (Population Stability Index)
    PSI < 0.1 → стабильно
    0.1 ≤ PSI < 0.2 → предупреждение
    PSI ≥ 0.2 → дрифт
    """
    # Объединяем для общих границ
    all_vals = reference_vals + current_vals
    if not all_vals:
        return 0.0
    
    min_val, max_val = min(all_vals), max(all_vals)
    if min_val == max_val:
        return 0.0
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    psi = 0.0
    for i in range(bins):
        ref_count = sum(1 for x in reference_vals if bin_edges[i] <= x < bin_edges[i+1])
        cur_count = sum(1 for x in current_vals if bin_edges[i] <= x < bin_edges[i+1])
        
        ref_pct = max(ref_count / len(reference_vals), 0.0001)
        cur_pct = max(cur_count / len(current_vals), 0.0001)
        
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    
    return psi


def get_actual_outcomes(user_ids, correct_dict):
    """
    Получить фактические ответы для списка пользователей
    
    Returns:
        list: список 1/0 для каждого пользователя (если данные есть)
    """
    outcomes = []
    for user_id in user_ids:
        try:
            df = st.load_csv(f"raw/users_logs/{user_id}.csv")
            if df.empty:
                continue
            
            # Берём последний ответ
            last_row = df.iloc[-1]
            user_answer = last_row.get("user_answer")
            question_id = last_row.get("question_id")
            
            if user_answer is None or question_id is None:
                continue
            
            correct_answer = correct_dict.get(question_id)
            if correct_answer is None:
                continue
            
            # 1 = правильно, 0 = неправильно
            outcomes.append(1 if user_answer == correct_answer else 0)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return outcomes


def load_correct_dict():
    """Загрузить словарь правильных ответов"""
    questions_df = st.load_csv("raw/contents/questions.csv")
    return dict(zip(questions_df["question_id"], questions_df["correct_answer"]))


def check_data_drift(current_features, reference_stats, feature_names):
    drift_report = {}
    max_psi = 0.0
    worst_feature = None
    
    # Преобразуем в numpy массив, если это список
    if not isinstance(current_features, np.ndarray):
        current_features = np.array(current_features, dtype=float)
    
    # Проверяем размерность
    if current_features.ndim == 1:
        # Если одномерный — превращаем в двумерный
        current_features = current_features.reshape(-1, 1)
    
    for i, feature_name in enumerate(feature_names):
        if i >= current_features.shape[1]:
            continue
        
        # Извлекаем значения и гарантируем плоский список чисел
        cur_vals = current_features[:, i].flatten()
        cur_vals = [float(x) for x in cur_vals]  # ← гарантируем числа
        
        ref_mean = reference_stats[feature_name]["mean"]
        ref_vals = [float(ref_mean)] * len(cur_vals)  # ← гарантируем числа
        
        psi = calculate_psi(ref_vals, cur_vals)
        
        status = "stable" if psi < PSI_WARNING else "warning" if psi < PSI_CRITICAL else "drift"
        
        drift_report[feature_name] = {
            "psi": round(psi, 4),
            "status": status,
            "current_mean": round(np.mean(cur_vals), 3),
            "reference_mean": round(ref_mean, 3)
        }
        
        if psi > max_psi:
            max_psi = psi
            worst_feature = feature_name
    
    return {
        "features": drift_report,
        "max_psi": round(max_psi, 4),
        "worst_feature": worst_feature,
        "overall_status": "drift" if max_psi >= PSI_CRITICAL else "warning" if max_psi >= PSI_WARNING else "stable"
    }


def check_target_drift(current_predictions, reference_mean):
    """
    Проверить target drift
    
    Args:
        current_predictions: list предсказаний (0-1)
        reference_mean: среднее предсказаний из эталона
    
    Returns:
        dict: {current_mean, reference_mean, current_positive_rate, drift_detected}
    """
    if not current_predictions:
        return {"status": "no_data", "message": "Нет предсказаний"}
    
    current_mean = np.mean(current_predictions)
    current_positive_rate = np.mean([1 if p > 0.5 else 0 for p in current_predictions])
    drift_detected = bool(abs(current_mean - reference_mean) > 0.1)
    
    return {
        "current_mean": round(current_mean, 4),
        "reference_mean": round(reference_mean, 4),
        "current_positive_rate": round(current_positive_rate, 3),
        "drift_detected": drift_detected,
        "status": "drift" if drift_detected else "stable"
    }


def check_concept_drift(current_predictions, user_ids, correct_dict, reference_accuracy=0.65):
    actual_outcomes = []
    preds_used = []
    
    for user_id, pred in zip(user_ids, current_predictions):
        try:
            df = st.load_csv(f"raw/users_logs/{user_id}.csv")
            if df.empty:
                continue
            
            respond_rows = df[df["action_type"] == "respond"]
            if respond_rows.empty:
                continue
            
            last_respond = respond_rows.iloc[-1]
            user_answer = last_respond.get("user_answer")
            item_id = last_respond.get("item_id")
            
            if user_answer is None or item_id is None:
                continue
            
            question_id = item_id
            
            correct_answer = correct_dict.get(question_id)
            if correct_answer is None:
                continue
            
            actual_outcomes.append(1 if user_answer == correct_answer else 0)
            preds_used.append(pred)
            
        except Exception as e:
            continue
    
    MIN_SAMPLES = 5
    if len(actual_outcomes) < MIN_SAMPLES:
        return {
            "status": "no_data",
            "message": f"Недостаточно данных: {len(actual_outcomes)} < {MIN_SAMPLES}",
            "samples_count": len(actual_outcomes)
        }
    
    correct = sum(1 for p, a in zip(preds_used, actual_outcomes) 
                  if (p > 0.5) == (a == 1))
    current_accuracy = correct / len(actual_outcomes)
    accuracy_drop = reference_accuracy - current_accuracy
    
    return {
        "current_accuracy": round(current_accuracy, 4),
        "reference_accuracy": round(reference_accuracy, 4),
        "accuracy_drop": round(accuracy_drop, 4),
        "status": "stable" if accuracy_drop < 0.05 else "warning" if accuracy_drop < 0.1 else "drift",
        "samples_count": len(actual_outcomes)
    }


def check_drift(window_size=100):
    """
    Полная проверка дрифта (data + target + concept)
    """
    print(f"[{datetime.now().isoformat()}] Проверка дрифта...")
    
    # 1. Загружаем эталон
    reference = load_reference()
    if reference is None:
        print("Нет эталона!")
        return None
    
    # 2. Загружаем текущее окно
    current_window = load_window(window_size)
    if len(current_window) < window_size:
        print(f"Недостаточно данных: {len(current_window)} < {window_size}")
        return None
    
    print(f"Окно: {len(current_window)} предсказаний")
    
    # 3. Подготовка данных
    feature_names = reference["feature_names"]
    ref_stats = reference["features"]
    current_features = np.array([w["features"] for w in current_window])
    current_predictions = [w["prediction"] for w in current_window]
    user_ids = [w["user_id"] for w in current_window]
    
    if current_features.size == 0:
        print("Нет признаков в окне")
        return None
    
    # 4. DATA DRIFT
    print("Data drift...")
    data_drift = check_data_drift(current_features, ref_stats, feature_names)
    
    # 5. TARGET DRIFT
    print("Target drift...")
    ref_mean = reference.get("target_stats", {}).get("mean", 0.5)
    target_drift = check_target_drift(current_predictions, ref_mean)
    
    # 6. CONCEPT DRIFT
    print("Concept drift...")
    correct_dict = load_correct_dict()
    ref_accuracy = reference.get("target_stats", {}).get("mean", 0.65)
    concept_drift = check_concept_drift(current_predictions, user_ids, correct_dict, ref_accuracy)
    
    # 7. Формируем отчёт
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "window_size": len(current_window),
        "data_drift": data_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
        "overall_status": data_drift["overall_status"]
    }
    
    # 8. Сохраняем
    drift_report_clean = clean_json(drift_report)
    st.save_json(drift_report_clean, "logs/drift/latest_report.json")
    if drift_report:
        text_report = generate_text_report(drift_report)
        
        # Сохраняем в MinIO
        st.save_text(text_report, "logs/drift/latest_text_report.txt")
        print("✅ Текстовый отчёт сохранён в MinIO")
    else:
        print("❌ Отчёт не сгенерирован")
    
    # История
    try:
        history = st.load_json("logs/drift/history.json")
    except Exception as e:
        print(f"Error: {e}")
        history = {"reports": []}
    history["reports"].append(drift_report_clean)
    history["reports"] = history["reports"][-100:]
    st.save_json(history, "logs/drift/history.json")
    
    status_emoji = "🟢" if data_drift["overall_status"] == "stable" else "🟡" if data_drift["overall_status"] == "warning" else "🔴"
    print(f"  {status_emoji} Data drift: {data_drift['overall_status']} (PSI={data_drift['max_psi']:.3f})")
    
    td = target_drift
    td_emoji = "🟢" if not td.get("drift_detected", False) else "🔴"
    print(f"  {td_emoji} Target drift: mean={td.get('current_mean', 0):.3f} vs {td.get('reference_mean', 0):.3f}")
    
    cd = concept_drift
    if cd.get("status") != "no_data":
        cd_emoji = "🟢" if cd["status"] == "stable" else "🟡" if cd["status"] == "warning" else "🔴"
        print(f"  {cd_emoji} Concept drift: {cd['status']} (acc={cd['current_accuracy']:.3f} vs {cd['reference_accuracy']:.3f})")
    else:
        print(f"Concept drift: нет данных ({cd.get('message', '')})")
    
    if data_drift["overall_status"] == "drift":
        print("Обнаружен дрифт данных! Нужно переобучение.")

        
    return drift_report


def generate_text_report(report):
    """Сгенерировать текстовый отчёт из JSON"""
    lines = []

    lines.append(" --- Data drift report ---")
    lines.append(f"Время: {report.get('timestamp', 'N/A')}")
    lines.append(f"Размер окна: {report.get('window_size', 'N/A')}")
    
    data_drift = report.get("data_drift", {})
    lines.append("\nDATA DRIFT:")
    lines.append(f"   Status: {data_drift.get('overall_status', 'unknown')}")
    lines.append(f"   Max PSI: {data_drift.get('max_psi', 0):.4f}")
    lines.append(f"   Worst feature: {data_drift.get('worst_feature', 'N/A')}")
    
    target_drift = report.get("target_drift", {})
    lines.append("\nTARGET DRIFT:")
    lines.append(f"   Current mean: {target_drift.get('current_mean', 0):.4f}")
    lines.append(f"   Reference mean: {target_drift.get('reference_mean', 0):.4f}")
    lines.append(f"   Drift: {'NO' if not target_drift.get('drift_detected', False) else 'YES'}")
    
    concept_drift = report.get("concept_drift", {})
    if concept_drift.get("status") != "no_data":
        lines.append("\nCONCEPT DRIFT:")
        lines.append(f"   Current accuracy: {concept_drift.get('current_accuracy', 0):.4f}")
        lines.append(f"   Referene accuracy: {concept_drift.get('reference_accuracy', 0):.4f}")
        lines.append(f"   Status: {concept_drift.get('status', 'unknown')}")
    
    print("\n"+"-"*50)
    if data_drift.get("overall_status") == "drift":
        lines.append("Reccomendation: drift detected! Finetune is recommended.")
    elif data_drift.get("overall_status") == "warning":
        lines.append("Reccomendation: drift`s signs detected. keep an eye on the model.")
    else:
        lines.append("Reccomendation: еhe data is stable, no retraining required.")
    print("\n"+"-"*50)
    
    return "\n".join(lines)


def clean_json(obj):
    """Рекурсивно преобразует объекты для JSON-сериализации"""
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_json(v) for v in obj)
    elif isinstance(obj, bool):
        return str(obj)  # bool → "true" / "false"
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    else:
        return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Проверка дрифта данных")
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        choices=[10, 100, 500, 1000],
        help="Размер скользящего окна (по умолчанию 100)"
    )
    args = parser.parse_args()
    report = check_drift(window_size=args.window)
    
    