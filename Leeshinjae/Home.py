import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="Spotify 고객 이탈 예측",
    page_icon="🎧",
    layout="wide"
)

# 사용자 정의 CSS
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}
.title {
    text-align: center;
    font-size: 26px;
    color: #ffffff;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 6px;
}
.headline-white {
    text-align: center;
    font-size: 96px;
    color: #ffffff;
    font-weight: 800;
    margin: 0;
    line-height: 1.2;
}
.headline-green {
    text-align: center;
    font-size: 96px;
    color: #1DB954;
    font-weight: 800;
    margin: 2px 0 12px 0;
    line-height: 1.2;
}
.arrow-button {
    display: flex;
    justify-content: center;
    margin-bottom: 24px;
}
.arrow-button button {
    font-size: 32px;
    color: #1DB954;
    background-color: transparent;
    border: none;
    cursor: pointer;
}
.description {
    color: #ffffff;
    text-align: center;
    font-size: 16px;
    margin: 8px auto 24px auto;
    max-width: 900px;
}
.metric {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-title {
    font-size: 24px;
    color: #1DB954;
    font-weight: 700;
    margin-bottom: 6px;
}
.metric-desc {
    font-size: 16px;
    color: #ffffff;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# 타이틀
st.markdown('<div class="title">🎧 Spotify Customer Analytics 🎵</div>', unsafe_allow_html=True)

# 가입 고객 / 이탈 예측
st.markdown('<div class="headline-white">가입 고객</div>', unsafe_allow_html=True)
st.markdown('<div class="headline-green">이탈 예측</div>', unsafe_allow_html=True)

# 화살표 버튼 (아래쪽에 배치)
st.markdown("""
<div class="arrow-button">
    <form action="/ChurnCheck" method="get">
        <button>➡️</button>
    </form>
</div>
""", unsafe_allow_html=True)

# 설명
st.markdown("""
<div class="description">
머신러닝과 딥러닝을 활용한 Spotify 고객 이탈 예측 모델 구축 및 배포 프로젝트입니다.<br>
고객 행동 데이터를 기반으로 이탈 가능성을 실시간으로 예측하여 비즈니스 전략 수립에 도움을 줍니다.
</div>
""", unsafe_allow_html=True)

# 메트릭 박스: 1행 4열
cols = st.columns(4)
metrics = [
    {"title": "ML/DL", "desc": "모델 활용"},
    {"title": "~95%+", "desc": "예측 정확도"},
    {"title": "4", "desc": "파이프라인 단계"},
    {"title": "Real-time", "desc": "배포 환경"}
]

for col, m in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="metric">
            <div class="metric-title">{m['title']}</div>
            <div class="metric-desc">{m['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.caption("© 2025 Spotify Churn Prediction Project")