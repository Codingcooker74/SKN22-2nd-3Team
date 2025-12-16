import streamlit as st

# 페이지 설정
st.set_page_config(page_title="이탈 예측 주요 피처", page_icon="📊", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# 상단 타이틀
st.markdown('<div class="key-features">Key Features</div>', unsafe_allow_html=True)
st.markdown("""
<div class="title-line">
    <span class="title-white">이탈 예측</span>
    <span class="title-green">주요 피처</span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="subtext">고객의 행동 패턴을 분석하여 이탈 가능성을 예측하는 핵심 지표들</div>', unsafe_allow_html=True)

# 1행 6열 박스 구성
cols = st.columns(6)
features = [
    {"icon": "⏰", "title": "이용 시간", "desc": "일별/주별/월별 청취 시간"},
    {"icon": "🎵", "title": "재생 패턴", "desc": "스킵률, 완청률 분석"},
    {"icon": "❤️", "title": "좋아요 활동", "desc": "저장 및 좋아요 빈도"},
    {"icon": "👥", "title": "소셜 활동", "desc": "공유 및 협업 플레이리스트"},
    {"icon": "📅", "title": "가입 기간", "desc": "구독 유지 기간 및 갱신 이력"},
    {"icon": "🎧", "title": "장르 다양성", "desc": "청취 장르 분포 및 변화"}
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

# 푸터
st.markdown("---")
st.caption("© 2025 Spotify Churn Prediction Project")