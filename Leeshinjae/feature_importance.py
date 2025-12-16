import streamlit as st
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="피처 중요도", page_icon="📊", layout="wide")

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
    font-size: 48px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 0px;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    color: #1DB954;
    margin-bottom: 40px;
}
.footer {
    text-align: center;
    font-size: 14px;
    color: #888888;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# 타이틀 및 서브타이틀
st.markdown('<div class="title">피처 중요도</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ML/DL 기반 고객 이탈 예측 프로젝트</div>', unsafe_allow_html=True)

# 피처 중요도 데이터
features = ["이용 시간", "재생 패턴", "가입 기간", "좋아요 활동", "장르 다양성", "소셜 활동"]
importance = [95, 88, 82, 75, 68, 55]

# 막대 그래프 생성
fig = go.Figure(go.Bar(
    x=features,
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

# 하단 설명
st.markdown('<div class="footer">Spotify Churn Prediction<br>© 2024 Customer Churn Prediction Project</div>', unsafe_allow_html=True)