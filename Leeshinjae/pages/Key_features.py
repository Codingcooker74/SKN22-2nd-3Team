import streamlit as st
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="Spotify 고객 이탈 예측 주요 피처 및 중요도", page_icon="📊", layout="wide")

# 사용자 정의 CSS
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Key Features: 작은 녹색 글자 */
.key-features {
    text-align: center;
    font-size: 20px;
    color: #1DB954;
    font-weight: bold;
    margin-bottom: 10px;
}

/* 타이틀 라인: 이탈 예측(흰색) + 주요 피처(녹색) */
.title-line {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    margin-bottom: 10px;
}
.title-white {
    color: #ffffff;
    display: inline;
}
.title-green {
    color: #1DB954;
    display: inline;
    margin-left: 12px;
}

/* 설명문 */
.subtext {
    text-align: center;
    font-size: 16px;
    color: #cccccc;
    margin-bottom: 40px;
}

/* 하단 피처 박스 */
.feature-box {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    padding: 16px;
    height: 160px;
    text-align: center;
}
.feature-icon {
    font-size: 28px;
    margin-bottom: 6px;
}
.feature-title {
    font-size: 18px;
    color: #1DB954;
    font-weight: bold;
    margin-bottom: 4px;
}
.feature-desc {
    font-size: 14px;
    color: #dddddd;
    line-height: 1.4;
}

/* 그래프 타이틀 */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #ffffff;
    margin-top: 60px;
    margin-bottom: 0px;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #1DB954;
    margin-bottom: 40px;
}

/* 네비게이션 버튼 스타일 (기존 디자인 계승) */
div[data-testid="stColumn"] div[data-testid="stButton"] > button {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    color: white;
    width: 100%;
    height: 60px;
    font-size: 18px;
    font-weight: bold;
    transition: all 0.3s ease;
}

div[data-testid="stColumn"] div[data-testid="stButton"] > button:hover {
    background-color: #1DB954;
    color: black;
}

/* 푸터 */
.footer {
    text-align: center;
    font-size: 14px;
    color: #888888;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 상단 타이틀 및 설명
st.markdown('<div class="key-features">Key Features</div>', unsafe_allow_html=True)
st.markdown("""
<div class="title-line">
    <span class="title-white">이탈 예측</span>
    <span class="title-green">주요 피처</span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="subtext">고객의 행동 패턴을 분석하여 이탈 가능성을 예측하는 핵심 지표들</div>', unsafe_allow_html=True)

# -------------------------------
# 주요 피처 박스 (1행 6열)
cols = st.columns(6)
features = [
    {"icon": "🎫", "title": "구독 타입", "desc": "무료/프리미엄/가족/스튜던트"},
    {"icon": "🎧", "title": "오프라인 재생시간", "desc": "오프라인 재생기능 사용"},
    {"icon": "📻", "title": "광고 청취시간", "desc": "주별 광고 청취 시간"},
    {"icon": "🎶", "title": "음악 재생시간", "desc": "일별 음악 재생 시간"},
    {"icon": "🌎", "title": "나라", "desc": "각 나라별"},
    {"icon": "😊", "title": "만족도 지수", "desc": "서비스 만족도 지수"}
]

for col, f in zip(cols, features):
    with col:
        st.markdown(f"""
        <div class="feature-box">
            <div class="feature-icon">{f['icon']}</div>
            <div class="feature-title">{f['title']}</div>
            <div class="feature-desc">{f['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------
# 피처 중요도 그래프
st.markdown('<div class="title">피처 중요도</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML/DL 기반 고객 이탈 예측 프로젝트</div>', unsafe_allow_html=True)

# 피처 중요도 데이터
features_importance = ["구독 타입", "오프라인 재생시간", "광고 청취시간", "음악 재생시간", "나라", "만족도 지수"]
importance = [95, 88, 82, 75, 68, 55]

# 막대 그래프 생성
fig = go.Figure(go.Bar(
    x=features_importance,
    y=importance,
    marker_color='#1DB954',
    text=[f"{v}%" for v in importance],
    textposition="outside"
))
fig.update_layout(
    plot_bgcolor="#000000",
    paper_bgcolor="#000000",
    font=dict(color="#ffffff"),
    xaxis=dict(title="피처", tickfont=dict(color="#ffffff")),
    yaxis=dict(title="중요도 (%)", tickfont=dict(color="#ffffff")),
    margin=dict(t=40, b=40, l=40, r=40),
    height=500
)

# 그래프 출력
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 하단 네비게이션 버튼 (1행 6열 구성을 통해 양 끝 라인 맞춤)
st.markdown("<br>", unsafe_allow_html=True)
nav_cols = st.columns(15)

with nav_cols[0]: # 첫 번째 박스 라인에 맞춤
    if st.button("🏠 Home"):
        st.switch_page("Home.py")

with nav_cols[14]: # 여섯 번째 박스 라인에 맞춤
    if st.button("Next ➡️"):
        st.switch_page("pages/model_comparison.py") # 다음 페이지 경로

# -------------------------------
# 푸터
st.markdown("---")
st.markdown('<div class="footer">Spotify Churn Prediction<br>© 2025 Customer Churn Prediction Project</div>', unsafe_allow_html=True)