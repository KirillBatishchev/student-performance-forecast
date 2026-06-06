import numpy as np
import pandas as pd


def Transform_questions_dict(questions_df: pd.DataFrame):
    correct_dict = dict(
        zip(questions_df["question_id"], questions_df["correct_answer"])
    )
    return correct_dict


def Transformation_data(df: pd.DataFrame, correct_dict: dict) -> pd.DataFrame:
    """
    Трансформация логов EdNET KT3 в датасет для обучения
    Одна запись на bundle
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    is_bundle_enter = df["item_id"].str.startswith("b", na=False) & (
        df["action_type"] == "enter"
    )
    is_bundle_submit = df["item_id"].str.startswith("b", na=False) & (
        df["action_type"] == "submit"
    )
    is_lecture_enter = df["item_id"].str.startswith("l", na=False) & (
        df["action_type"] == "enter"
    )
    is_lecture_quit = df["item_id"].str.startswith("l", na=False) & (
        df["action_type"] == "quit"
    )
    is_explanation_enter = df["item_id"].str.startswith("e", na=False) & (
        df["action_type"] == "enter"
    )
    is_explanation_quit = df["item_id"].str.startswith("e", na=False) & (
        df["action_type"] == "quit"
    )
    is_respond = df["item_id"].str.startswith("q", na=False) & (
        df["action_type"] == "respond"
    )

    # Собираем bundle
    bundle_starts = df[is_bundle_enter][
        ["item_id", "timestamp", "platform", "source"]
    ].rename(columns={"timestamp": "enter_time"})
    bundle_ends = df[is_bundle_submit][["item_id", "timestamp"]].rename(
        columns={"timestamp": "submit_time"}
    )
    bundles = bundle_starts.merge(bundle_ends, on="item_id", how="inner")
    if len(bundles) == 0:
        return pd.DataFrame()
    bundles = (
        bundles.groupby("item_id")
        .agg(
            {
                "enter_time": "min",
                "submit_time": "max",
                "platform": "first",
                "source": "first",
            }
        )
        .reset_index()
    )

    bundles = bundles[bundles["submit_time"] > bundles["enter_time"]].copy()

    # Lectures
    lectures_enter = df[is_lecture_enter][["item_id", "timestamp"]].rename(
        columns={"timestamp": "enter_time"}
    )
    lectures_quit = df[is_lecture_quit][["item_id", "timestamp"]].rename(
        columns={"timestamp": "quit_time"}
    )

    # Responds
    responds = df[is_respond][["item_id", "timestamp", "user_answer"]]

    # Adding temporary indexes for sequential matching
    lectures_enter["n"] = lectures_enter.groupby("item_id").cumcount()
    lectures_quit["n"] = lectures_quit.groupby("item_id").cumcount()
    lectures = lectures_enter.merge(
        lectures_quit, on=["item_id", "n"], how="inner"
    )
    lectures = lectures.drop("n", axis=1)

    # Explanations
    explanations_enter = df[is_explanation_enter][
        ["item_id", "timestamp"]
    ].rename(columns={"timestamp": "enter_time"})
    explanations_quit = df[is_explanation_quit][
        ["item_id", "timestamp"]
    ].rename(columns={"timestamp": "quit_time"})

    # Adding temporary indexes for sequential matching
    explanations_enter["n"] = explanations_enter.groupby("item_id").cumcount()
    explanations_quit["n"] = explanations_quit.groupby("item_id").cumcount()
    explanations = explanations_enter.merge(
        explanations_quit, on=["item_id", "n"], how="inner"
    )
    explanations = explanations.drop("n", axis=1)

    bundles = bundles.sort_values("enter_time").reset_index(drop=True)

    results = []

    accuracy_history = []

    for idx, row in bundles.iterrows():
        enter_time = row["enter_time"]
        submit_time = row["submit_time"]
        platform = row.platform
        source = row.source

        # Bundle's time
        bundle_time_limited = min(submit_time - enter_time, 600000)

        if idx == 0:
            start_interval = enter_time
        else:
            start_interval = bundles.iloc[idx - 1]["submit_time"]

        end_interval = submit_time

        if len(lectures) > 0:
            lecture_mask = (
                (lectures["enter_time"] >= start_interval)
                & (lectures["quit_time"] <= end_interval)
                & (lectures["enter_time"] < lectures["quit_time"])
            )
            lectures_count = lecture_mask.sum()
        else:
            lectures_count = 0

        if len(explanations) > 0:
            exp_mask = (
                (explanations["enter_time"] >= start_interval)
                & (explanations["quit_time"] <= end_interval)
                & (explanations["enter_time"] < explanations["quit_time"])
            )
            explanations_count = exp_mask.sum()
        else:
            explanations_count = 0

        mask = (responds["timestamp"] > start_interval) & (
            responds["timestamp"] < end_interval
        )
        bundle_responds = responds[mask].copy()

        n_questions = 0
        n_attempts = 0
        accuracy = 0
        n_correct = 0

        if len(bundle_responds) > 0:
            n_questions = bundle_responds["item_id"].nunique()
            n_attempts = len(bundle_responds)

            for qid, group in bundle_responds.groupby("item_id"):
                group_sorted = group.sort_values("timestamp")
                last_answer = group_sorted.iloc[-1]["user_answer"]

                if last_answer == correct_dict.get(qid, ""):
                    n_correct += 1

            accuracy = n_correct / n_questions if n_questions > 0 else 0

        # Time (sin/cos)
        dt = pd.to_datetime(submit_time, unit="ms")
        hour = dt.hour
        hour_rad = 2 * np.pi * hour / 24
        hour_sin = np.sin(hour_rad)
        hour_cos = np.cos(hour_rad)

        # Day of week (sin/cos)
        day_of_week = dt.dayofweek  # 0=mon, 6=sun
        dow_rad = 2 * np.pi * day_of_week / 7
        dow_sin = np.sin(dow_rad)
        dow_cos = np.cos(dow_rad)

        # Weekend flag
        is_weekend = 1 if day_of_week >= 5 else 0

        # Time between bundles
        if idx == 0:
            time_since_last = 0
        else:
            prev_submit = bundles.iloc[idx - 1]["submit_time"]
            time_since_last = (submit_time - prev_submit) / 1000 / 60
            time_since_last = max(0, min(time_since_last, 240))

        # Moving average accuracy
        accuracy_history.append(accuracy)

        if len(accuracy_history) >= 2:
            ma3 = np.mean(
                accuracy_history[-min(3, len(accuracy_history) - 1) : -1]
            )
        else:
            ma3 = 0

        # Accuracy trend
        if len(accuracy_history) < 2:
            accuracy_trend = 0
        else:
            accuracy_trend = accuracy_history[-2]

        results.append(
            {
                "timestamp": submit_time,
                "hour_sin": round(hour_sin, 6),
                "hour_cos": round(hour_cos, 6),
                "dow_sin": round(dow_sin, 6),
                "dow_cos": round(dow_cos, 6),
                "is_weekend": is_weekend,
                "time_since_last_min": round(time_since_last, 2),
                "platform": platform,
                "source": source,
                "bundle_time_ms": bundle_time_limited,
                "viewed_lectures": lectures_count,
                "viewed_explanations": explanations_count,
                "n_questions": n_questions,
                "n_attempts": n_attempts,
                "n_correct": n_correct,
                "accuracy": round(accuracy, 3),
                "accuracy_ma3": round(ma3, 6),
                "accuracy_trend": round(accuracy_trend, 6),
            }
        )

    if results:
        df = (
            pd.DataFrame(results)
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        df["accuracy_ma3"] = df["accuracy_ma3"].fillna(0)
        df["accuracy_trend"] = df["accuracy_trend"].fillna(0)
        return df
    return pd.DataFrame()
