import streamlit as st

# 페이지 설정
st.set_page_config(page_title="프로젝트 파이프라인", page_icon="🧠", layout="wide")

# 사용자 정의 CSS
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    text-align: center;
    color: #1DB954;
    font-size: 48px;
    margin-bottom: 10px;
    letter-spacing: 2px; /* 글자 간격 */
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
    margin-bottom: 40px;
    letter-spacing: 1px; /* 글자 간격 */
}
.pipeline-card {
    background-color: #111111;
    border: 1px solid #1DB954;
    border-radius: 12px;
    padding: 20px;
    height: 320px;
}
.pipeline-title {
    font-size: 22px;
    color: #1DB954;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
    letter-spacing: 1px; /* 글자 간격 */
}
.pipeline-icon {
    font-size: 40px;
    text-align: center;
    margin-bottom: 8px;
}
.pipeline-subtitle {
    font-size: 16px;
    color: #ffffff;
    text-align: center;
    margin-bottom: 12px;
    letter-spacing: 1px; /* 글자 간격 */
}
.pipeline-list {
    font-size: 14px;
    color: #dddddd;
    margin-left: 10px;
    line-height: 1.6; /* 줄 간격 */
    letter-spacing: 0.5px; /* 글자 간격 */
}
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown("<h1>프로젝트 파이프라인</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">체계적인 4단계 프로세스를 통해 고객 이탈을 정확하게 예측합니다</div>', unsafe_allow_html=True)

# 단계별 정보
pipeline_steps = [
    {
        "icon": "📊",
        "title": "01. 비즈니스 이해",
        "subtitle": "머신러닝 모델 활용 계획 수립",
        "items": [
            "이탈의 비즈니스 영향 분석",
            "KPI 및 성공 지표 정의",
            "데이터 요구사항 파악",
            "프로젝트 범위 및 목표 설정"
        ]
    },
    {
        "icon": "🧹",
        "title": "02. 데이터 준비",
        "subtitle": "데이터 셋 준비 및 전처리",
        "items": [
            "데이터 수집 및 통합",
            "결측치 및 이상치 처리",
            "피처 엔지니어링",
            "데이터 정규화 및 인코딩"
        ]
    },
    {
        "icon": "🤖",
        "title": "03. 모델 학습",
        "subtitle": "ML 및 DL 모델 학습과 평가",
        "items": [
            "다양한 알고리즘 실험",
            "하이퍼파라미터 튜닝",
            "교차 검증 수행",
            "모델 성능 비교 분석"
        ]
    },
    {
        "icon": "🚀",
        "title": "04. 모델 배포",
        "subtitle": "최적 모델 설정 및 배포",
        "items": [
            "최종 모델 선정",
            "API 엔드포인트 구축",
            "모니터링 시스템 설정",
            "실시간 예측 서비스 배포"
        ]
    }
]

# 1행 4열 구성
cols = st.columns(4)
for col, step in zip(cols, pipeline_steps):
    with col:
        st.markdown(f"""
        <div class="pipeline-card">
            <div class="pipeline-title">{step['title']}</div>
            <div class="pipeline-icon">{step['icon']}</div>
            <div class="pipeline-subtitle">{step['subtitle']}</div>
            <ul class="pipeline-list">
                {''.join(f"<li>{item}</li>" for item in step['items'])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

# 푸터
st.markdown("---")
st.caption("© 2025 Spotify Churn Prediction Project")