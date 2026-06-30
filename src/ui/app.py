import streamlit as st
import plotly.express as px
import requests

API_URL = "http://student-performance-forecast-service"
# API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Student Performance Forecast",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e2d3d;
        border: 1px solid #2d6a9f;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .success-badge {
        background: #1a472a;
        color: #51cf66;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .risk-badge {
        background: #4a1942;
        color: #ff6b6b;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Шапка
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🎓 Student Performance Forecast</h1>
    <p style="color: #e0e0e0; margin: 5px 0 0 0;">MLOps Dashboard — мониторинг и управление моделью предсказания успеваемости</p>
</div>
""", unsafe_allow_html=True)

# Сайдбар
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
    st.title("Навигация")
    page = st.radio(
        "",
        ["Предсказание", "История предсказаний", "Эксперименты", "Дрейф данных"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Student Performance Forecast v1.0")

# ─── Инференс ─────────────────────────────────────────────────
if page == "Предсказание":
    st.subheader("Предсказание успеваемости")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_ids = st.text_area(
            "ID пользователей (каждый с новой строки)",
            placeholder="u10013\nu1020\nu10200",
            height=150
        )
    with col2:
        st.info("**Как использовать:**\n\nВведите ID студентов по одному на строке и нажмите кнопку предсказания.")

    if st.button("Получить предсказание", type="primary", use_container_width=True):
        if user_ids:
            ids = [uid.strip() for uid in user_ids.strip().split("\n") if uid.strip()]
            with st.spinner("Выполняется инференс..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"user_ids": ids},
                        timeout=60
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Запрос выполнен | ID: `{data['request_id']}` | Время: {data['timestamp']}")

                        st.divider()
                        for pred in data["predictions"]:
                            if "error" in pred:
                                st.error(f"❌ **{pred['user_id']}**: {pred['error']}")
                            else:
                                with st.container():
                                    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
                                    c1.metric("User ID", pred["user_id"])
                                    prob = pred["prediction"]
                                    c2.metric("Вероятность", f"{prob:.1%}")
                                    c3.progress(prob)
                                    if pred["will_succeed"]:
                                        c4.markdown(
                                            '<span class="success-badge">✅ Успех</span>', unsafe_allow_html=True)
                                    else:
                                        c4.markdown('<span class="risk-badge">⚠️ Риск</span>', unsafe_allow_html=True)
                    else:
                        st.error(f"Ошибка API: {response.status_code}")
                except Exception as e:
                    st.error(f"Ошибка подключения: {e}")
        else:
            st.warning("⚠️ Введите хотя бы один ID пользователя")

# ─── Предсказания ─────────────────────────────────────────────
elif page == "История предсказаний":
    st.subheader("История предсказаний")

    # --- МЕТРИКИ ПРЕДСКАЗАНИЙ ---
    try:
        response = requests.get(f"{API_URL}/predictions/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Всего", data.get("total", 0))
            col2.metric("✅ Успех", data.get("success_count", 0))
            col3.metric("❌ Риск провала", data.get("failure_count", 0))
            col4.metric("Средняя уверенность", f"{data.get('avg_confidence', 0):.1%}")
        else:
            st.warning("Статистика предсказаний недоступна")
    except BaseException:
        pass

    st.divider()

    # Фильтры
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        limit = st.selectbox("Количество записей", options=[10, 20, 50, 100], index=2)
    with col2:
        show_anomalies = st.checkbox("🔴 Только аномалии", value=False)
    with col3:
        st.caption("Аномалия: вероятность < 30% или > 70%")

    try:
        # --- ИСПОЛЬЗУЕМ /predictions/history ---
        response = requests.get(
            f"{API_URL}/predictions/history?limit={limit * 2}",
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            predictions = data.get("predictions", [])

            # Функция проверки аномалии
            def is_anomaly(p):
                prob = p.get("prediction", 0.5)
                return prob < 0.3 or prob > 0.7

            # Фильтр
            if show_anomalies:
                predictions = [p for p in predictions if is_anomaly(p)]

            if predictions:
                rows = []
                for p in predictions:
                    if "error" in p:
                        rows.append({
                            "User ID": p["user_id"],
                            "Вероятность": "-",
                            "Статус": "❌ Ошибка",
                            "Аномалия": "❌",
                            "Время": p.get("timestamp", "")[:19] if p.get("timestamp") else "",
                            "Версия": p.get("model_version", "N/A")
                        })
                    else:
                        prob = p.get("prediction", 0)
                        anomaly = is_anomaly(p)
                        rows.append({
                            "User ID": p["user_id"],
                            "Вероятность": f"{prob:.1%}",
                            "Статус": "✅ Успех" if p.get("will_succeed") else "⚠️ Риск",
                            "Аномалия": "🔴" if anomaly else "🟢",
                            "Время": p.get("timestamp", "")[:19] if p.get("timestamp") else "",
                            "Версия": p.get("model_version", "N/A")
                        })

                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"Показано {len(rows)} предсказаний" + (" (только аномалии)" if show_anomalies else ""))
            else:
                st.info("Нет предсказаний" + (" с аномалиями" if show_anomalies else ""))
        else:
            st.error(f"Ошибка API: {response.status_code}")
    except Exception as e:
        st.error(f"Ошибка подключения: {e}")

# ─── Эксперименты ─────────────────────────────────────────────
elif page == "Эксперименты":
    st.subheader("Эксперименты MLflow")
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            st.success("✅ API доступен")
        else:
            st.error(f"❌ API недоступен: {response.status_code}")
    except Exception as e:
        st.error(f"❌ API недоступен: {e}")
    # Кнопки управления
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Обновить эксперименты", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Переобучение", type="primary", use_container_width=True):
            with st.spinner("Запускаем переобучение..."):
                try:
                    response = requests.post(f"{API_URL}/finetune", timeout=600)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Запущено! Job ID: `{data['job_id']}`")
                    else:
                        st.error(f"Ошибка: {response.status_code}")
                except Exception as e:
                    st.error(f"Ошибка: {e}")
    with col3:
        if st.button("Полное обучение", type="secondary", use_container_width=True):
            if st.checkbox("⚠️ Подтверждаю запуск полного обучения (5-15 мин)"):
                with st.spinner("Запускаем полное обучение..."):
                    try:
                        response = requests.post(f"{API_URL}/train", timeout=600)
                        if response.status_code == 200:
                            st.success("✅ Обучение запущено!")
                        else:
                            st.error(f"Ошибка: {response.status_code}")
                    except Exception as e:
                        st.error(f"Ошибка: {e}")

    st.divider()

    # Таблица экспериментов
    try:
        response = requests.get(f"{API_URL}/experiments", timeout=30)
        if response.status_code == 200:
            data = response.json()
            experiments = data.get("experiments", [])

            if experiments:
                rows = []
                for exp in experiments:
                    rows.append({
                        "Название": exp.get("name", "N/A"),
                        "ID": exp.get("id", "N/A"),
                        "Статус": exp.get("status", "N/A"),
                        "Runs": exp.get("runs", 0)
                    })

                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"Всего экспериментов: {len(rows)}")
            else:
                st.info("Нет экспериментов")
        else:
            st.error(f"Ошибка загрузки экспериментов: {response.status_code}")
    except Exception as e:
        st.error(f"Ошибка подключения: {e}")

    st.divider()

    # --- СРАВНЕНИЕ ВЕРСИЙ МОДЕЛЕЙ ---
    st.subheader("Сравнение версий моделей")

    try:
        response = requests.get(f"{API_URL}/model/metrics", timeout=30)
        if response.status_code == 200:
            data = response.json()
            versions = data.get("versions", [])

            if versions:
                rows = []
                for v in versions:
                    rows.append({
                        "Версия": v.get("version", "N/A"),
                        "Accuracy": f"{v.get('accuracy', 0):.3f}",
                        "Loss": f"{v.get('loss', 0):.3f}",
                        "Дата": v.get("timestamp", "N/A")[:19] if v.get("timestamp") else "N/A"
                    })

                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.caption(f"Всего версий: {len(rows)}")

                # График сравнения Accuracy
                try:
                    df_plot = pd.DataFrame(rows)
                    df_plot["Версия"] = df_plot["Версия"].astype(str)
                    df_plot["Accuracy"] = df_plot["Accuracy"].astype(float)

                    fig = px.line(
                        df_plot,
                        x="Версия",
                        y="Accuracy",
                        title="Accuracy по версиям моделей",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.caption(f"График недоступен: {e}")

                # Текущая версия
                if versions:
                    latest = versions[-1]
                    st.success(
                        f"Текущая версия: **{latest.get('version', 'N/A')}** (Accuracy: {latest.get('accuracy', 0):.3f})")

            else:
                st.info("Нет данных о метриках моделей")
        else:
            st.warning("Метрики моделей недоступны")
    except Exception as e:
        st.warning(f"Метрики моделей недоступны: {e}")

# ─── Дрейф ────────────────────────────────────────────────────
elif page == "Дрейф данных":
    st.subheader("Мониторинг дрейфа и версия модели")

    # ============================================
    # КЕШИРОВАННЫЕ ЗАПРОСЫ
    # ============================================
    @st.cache_data(ttl=30)
    def get_cached_model_version():
        try:
            response = requests.get(f"{API_URL}/model/version", timeout=5)
            if response.status_code == 200:
                return response.json().get("version", "unknown")
        except BaseException:
            pass
        return "unknown"

    @st.cache_data(ttl=30)
    def get_cached_data_info():
        try:
            response = requests.get(f"{API_URL}/data/info", timeout=5)
            if response.status_code == 200:
                return response.json()
        except BaseException:
            pass
        return {"total_users": 0, "new_users": 0, "total_predictions": 0}

    @st.cache_data(ttl=30)
    def get_cached_drift_status():
        try:
            response = requests.get(f"{API_URL}/drift/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except BaseException:
            pass
        return {"status": "no_data", "max_psi": 0}

    # ============================================
    # ОТОБРАЖЕНИЕ
    # ============================================

    # Версия модели
    model_version = get_cached_model_version()
    if model_version and model_version != "unknown":
        st.metric("Версия модели", f"v{model_version}")
    else:
        st.metric("Версия модели", "⚠️ недоступна")

    st.divider()

    # Информация о данных
    st.subheader("Информация о данных")
    data_info = get_cached_data_info()

    if data_info.get("total_users", 0) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Всего пользователей", data_info.get("total_users", 0))
        col2.metric("Всего предсказаний", data_info.get("total_predictions", 0))
        col3.metric("Новых пользователей", data_info.get("new_users", 0))
    else:
        st.warning("Информация о данных временно недоступна")
        if st.button("Попробовать снова"):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # Выбор размера окна
    col1, col2 = st.columns([1, 3])
    with col1:
        window_size = st.selectbox(
            "Размер окна для детекции дрифта",
            options=[10, 50, 100, 250, 500, 1000],
            index=3,
            help="Размер скользящего окна для расчёта PSI"
        )
    with col2:
        st.caption("Чем больше окно, тем стабильнее показатель PSI. Рекомендуется 100-250.")

    # Кнопка проверки дрифта
    if st.button("Проверить дрифт", type="primary", use_container_width=True):
        with st.spinner(f"Проверка дрифта (окно={window_size})..."):
            try:
                response = requests.post(
                    f"{API_URL}/check_drift",
                    params={"window_size": window_size},
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Проверка дрифта выполнена")
                    st.cache_data.clear()  # Очищаем кеш после проверки

                    # --- ОБЩИЙ СТАТУС ---
                    status = data.get("overall_status", "unknown")
                    st.subheader("Общий статус")

                    if status == "drift":
                        st.error("🔴 **ДРИФТ ОБНАРУЖЕН!** Рекомендуется переобучение.")
                    elif status == "warning":
                        st.warning("🟡 **Предупреждение:** обнаружены признаки дрифта.")
                    elif status == "stable":
                        st.success("🟢 **Стабильно:** дрифта не обнаружено.")
                    else:
                        st.info("Нет данных о дрифте")

                    # --- МЕТРИКИ ДРИФТА ---
                    st.divider()
                    st.subheader("Метрики дрифта")

                    data_drift = data.get("data_drift", {})
                    target_drift = data.get("target_drift", {})
                    concept_drift = data.get("concept_drift", {})

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        max_psi = data_drift.get("max_psi", 0)
                        st.metric(
                            "Data Drift (PSI)",
                            f"{max_psi:.3f}",
                            help="PSI > 0.2 — дрифт данных"
                        )
                        worst_feature = data_drift.get("worst_feature", "—")
                        st.caption(f"Худший признак: `{worst_feature}`")

                    with col2:
                        target_status = target_drift.get("status", "no_data")
                        current_mean = target_drift.get("current_mean", 0)
                        ref_mean = target_drift.get("reference_mean", 0)
                        target_emoji = "🟢" if target_status == "stable" else "🔴" if target_status == "drift" else "⚪"
                        st.metric(
                            "Target Drift",
                            f"{target_emoji} {target_status}",
                            delta=f"{current_mean:.3f} vs {ref_mean:.3f}",
                            help="Изменение распределения целевой переменной"
                        )

                    with col3:
                        concept_status = concept_drift.get("status", "no_data")
                        current_acc = concept_drift.get("current_accuracy", 0)
                        ref_acc = concept_drift.get("reference_accuracy", 0)
                        concept_emoji = "🟢" if concept_status == "stable" else "🔴" if concept_status == "drift" else "⚪"
                        st.metric(
                            "Concept Drift",
                            f"{concept_emoji} {concept_status}",
                            delta=f"{current_acc:.3f} vs {ref_acc:.3f}" if concept_status != "no_data" else "нет данных",
                            help="Изменение точности модели")

                    # --- ДЕТАЛЬНЫЙ ОТЧЁТ ПО ПРИЗНАКАМ ---
                    st.divider()
                    st.subheader("Детальный отчёт по признакам")

                    features = data_drift.get("features", {})
                    if features:
                        rows = []
                        for feature_name, stats in features.items():
                            psi = stats.get("psi", 0)
                            status = stats.get("status", "unknown")
                            current_mean = stats.get("current_mean", 0)
                            ref_mean = stats.get("reference_mean", 0)

                            status_emoji = "🟢" if status == "stable" else "🟡" if status == "warning" else "🔴"
                            rows.append({
                                "Признак": feature_name,
                                "PSI": f"{psi:.3f}",
                                "Статус": f"{status_emoji} {status}",
                                "Текущее среднее": f"{current_mean:.3f}",
                                "Эталонное среднее": f"{ref_mean:.3f}"
                            })

                        import pandas as pd
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        # Сводка по признакам
                        total_features = len(rows)
                        drift_count = sum(1 for r in rows if "drift" in r["Статус"])
                        warning_count = sum(1 for r in rows if "warning" in r["Статус"])
                        stable_count = total_features - drift_count - warning_count

                        c1, c2, c3 = st.columns(3)
                        c1.metric("🔴 Дрифт", drift_count)
                        c2.metric("🟡 Предупреждение", warning_count)
                        c3.metric("🟢 Стабильно", stable_count)

                        # Предупреждение о критических признаках
                        if drift_count > 0:
                            critical_features = [r["Признак"] for r in rows if "drift" in r["Статус"]]
                            st.warning(
                                f"⚠️ **{drift_count}** признаков с обнаруженным дрифтом: {', '.join(critical_features)}")
                    else:
                        st.info("Нет данных о признаках")

                else:
                    st.error(f"Ошибка проверки дрифта: {response.status_code}")
            except Exception as e:
                st.error(f"Ошибка: {e}")

    # --- АВТОМАТИЧЕСКОЕ ОБНОВЛЕНИЕ СТАТУСА ---
    st.divider()
    st.subheader("Текущий статус дрифта")

    drift_status = get_cached_drift_status()
    status = drift_status.get("status", "no_data")
    psi = drift_status.get("max_psi", 0)
    timestamp = drift_status.get("timestamp", "")

    col1, col2 = st.columns(2)
    with col1:
        if status == "drift":
            st.error(f"🔴 Статус: **ДРИФТ** (PSI: {psi:.3f})")
        elif status == "warning":
            st.warning(f"🟡 Статус: **ПРЕДУПРЕЖДЕНИЕ** (PSI: {psi:.3f})")
        elif status == "stable":
            st.success(f"🟢 Статус: **СТАБИЛЬНО** (PSI: {psi:.3f})")
        else:
            st.info("Статус: нет данных")

    with col2:
        st.caption(f"Обновлено: {timestamp[:19] if timestamp else '—'}")

    # Кнопка обновления статуса
    if st.button("Обновить статус"):
        st.cache_data.clear()
        st.rerun()
