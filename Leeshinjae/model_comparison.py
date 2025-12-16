import streamlit as st

# 페이지 설정
st.set_page_config(page_title="모델 성능 비교", page_icon="📊", layout="wide")

# 사용자 정의 CSS
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}
.title-white {
    font-size: 48px;
    color: #ffffff;
    font-weight: bold;
    margin-left: 20px;
    line-height: 1.2;
}
.title-green {
    font-size: 48px;
    color: #1DB954;
    font-weight: bold;
    margin-left: 20px;
    margin-bottom: 30px;
    line-height: 1.2;
}
.small-box {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    height: 120px;
}
.small-title {
    font-size: 18px;
    color: #1DB954;
    font-weight: bold;
    margin-bottom: 6px;
}
.small-value {
    font-size: 24px;
    color: #ffffff;
    font-weight: bold;
}
.large-box {
    background-color: #1c1c1c;
    border: 1px solid #1DB954;
    border-radius: 12px;
    padding: 16px;
    height: 180px; /* 기존보다 20% 줄임 */
    margin: 12px;
}
.large-title {
    font-size: 18px;
    color: #1DB954;
    font-weight: bold;
    margin-bottom: 6px;
}
.large-icon {
    font-size: 28px;
    margin-bottom: 6px;
}
.large-desc {
    font-size: 13px;
    color: #dddddd;
    margin-bottom: 10px;
}
.large-score {
    font-size: 18px;
    color: #ffffff;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# 전체 레이아웃: 좌측(타이틀 + 하단 박스) / 우측(모델 박스)
left_col, right_col = st.columns([1, 1])  # 균등 분할 → 우측 박스가 화면 가운데부터 시작

# 좌측 타이틀
with left_col:
    st.markdown('<div class="title-white">다양한 모델</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-green">성능 비교</div>', unsafe_allow_html=True)

# 우측 모델 박스: 2행 2열
with right_col:
    models = [
        {"icon": "🌲", "title": "Random Forest", "desc": "양상을 기반 의사결정 트리 모델", "score": "92.3%"},
        {"icon": "⚡", "title": "XGBoost", "desc": "그래디언트 부스팅 알고리즘", "score": "94.7%"},
        {"icon": "🧠", "title": "Neural Network", "desc": "다층 퍼셉트론 딥러닝 모델", "score": "95.2%"},
        {"icon": "📈", "title": "LSTM", "desc": "시계열 패턴 학습 RNN 모델", "score": "96.1%"}
    ]
    for i in range(0, len(models), 2):
        row = st.columns([1, 1])
        for col, model in zip(row, models[i:i+2]):
            with col:
                st.markdown(f"""
                <div class="large-box">
                    <div class="large-icon">{model['icon']}</div>
                    <div class="large-title">{model['title']}</div>
                    <div class="large-desc">{model['desc']}</div>
                    <div class="large-score">성능: {model['score']}</div>
                </div>
                """, unsafe_allow_html=True)

# 좌측 하단 성능 지표 박스 (우측 박스 하단과 수평 맞춤)
with left_col:
    st.markdown("<br><br>", unsafe_allow_html=True)  # 여백으로 수평 맞춤
    st.subheader("📊 핵심 성능 지표")
    small_cols = st.columns(3)
    metrics = [
        {"title": "Accuracy", "value": "96.1%"},
        {"title": "Precision", "value": "94.8%"},
        {"title": "F1 Score", "value": "95.4%"}
    ]
    for col, m in zip(small_cols, metrics):
        with col:
            st.markdown(f"""
            <div class="small-box">
                <div class="small-title">{m['title']}</div>
                <div class="small-value">{m['value']}</div>
            </div>
            """, unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.caption("© 2025 Spotify Churn Prediction Project")