import streamlit as st
import requests

API_URL = "http://130.49.153.56:8888"

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
        ["🔮 Инференс", "📋 Предсказания", "🧪 Эксперименты", "📊 Дрейф"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Student Performance Forecast v1.0")

# ─── Инференс ─────────────────────────────────────────────────
if page == "🔮 Инференс":
    st.subheader("🔮 Предсказание успеваемости")

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
                                    c1.metric("👤 User ID", pred["user_id"])
                                    prob = pred["prediction"]
                                    c2.metric("📊 Вероятность", f"{prob:.1%}")
                                    c3.progress(prob)
                                    if pred["will_succeed"]:
                                        c4.markdown('<span class="success-badge">✅ Успех</span>', unsafe_allow_html=True)
                                    else:
                                        c4.markdown('<span class="risk-badge">⚠️ Риск</span>', unsafe_allow_html=True)
                    else:
                        st.error(f"Ошибка API: {response.status_code}")
                except Exception as e:
                    st.error(f"Ошибка подключения: {e}")
        else:
            st.warning("⚠️ Введите хотя бы один ID пользователя")

# ─── Предсказания ─────────────────────────────────────────────
elif page == "📋 Предсказания":
    st.subheader("📋 Последние предсказания")
    st.info("История предсказаний будет доступна после подключения БД")

# ─── Эксперименты ─────────────────────────────────────────────
elif page == "🧪 Эксперименты":
    st.subheader("🧪 Эксперименты MLflow")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Запустить переобучение", type="primary", use_container_width=True):
            with st.spinner("Запускаем переобучение..."):
                try:
                    response = requests.post(f"{API_URL}/retrain", timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Запущено! Job ID: `{data['job_id']}`")
                    else:
                        st.error(f"Ошибка: {response.status_code}")
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    st.info("Таблица экспериментов MLflow будет добавлена после настройки API эндпоинта /experiments")

# ─── Дрейф ────────────────────────────────────────────────────
elif page == "📊 Дрейф":
    st.subheader("📊 Мониторинг дрейфа")

    col1, col2, col3 = st.columns(3)
    col1.metric("Data Drift", "Нет данных", help="Дрейф входных данных")
    col2.metric("Target Drift", "Нет данных", help="Дрейф целевой переменной")
    col3.metric("Concept Drift", "Нет данных", help="Концептуальный дрейф")

    st.info("Мониторинг дрейфа будет доступен после настройки Evidently")
