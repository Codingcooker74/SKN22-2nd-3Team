import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# 1. 설정 및 리소스 로드
# ---------------------------------------------------------
st.set_page_config(
    page_title="Spotify Churn Predictor",
    page_icon="🎵",
    layout="centered"
)

# 스타일 커스텀 (스포티파이 테마)
st.markdown(
    """
    <style>
    .main {
        background-color: #121212; /* Spotify Dark BG */
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #1DB954; /* Spotify Green */
        color: white;
        border-radius: 20px;
        border: none;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #1DB954 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model_pipeline():
    # 저장된 전체 파이프라인(전처리 + 모델)을 불러옵니다.
    try:
        pipeline = joblib.load('models/spotify_churn_model.pkl')
        return pipeline
    except FileNotFoundError:
        st.error("모델 파일('models/spotify_churn_model.pkl')을 찾을 수 없습니다. 먼저 모델 학습을 완료해주세요.")
        return None

pipeline = load_model_pipeline()

# ---------------------------------------------------------
# 2. 헤더 섹션
# ---------------------------------------------------------
st.title("🎵 Spotify 유저 이탈 예측 시스템")
st.write("유저의 활동 데이터를 기반으로 다음 달 구독 해지(Churn) 가능성을 예측합니다.")
st.markdown("---")

# ---------------------------------------------------------
# 3. 입력 폼 섹션 (사이드바 또는 메인)
# ---------------------------------------------------------
st.header("📋 유저 정보 입력")

col1, col2 = st.columns(2)

with col1:
    st.subheader("기본 정보")
    age = st.slider("나이 (Age)", 18, 70, 30)
    gender = st.selectbox("성별 (Gender)", ['Male', 'Female', 'Other'])
    country = st.selectbox("국가 (Country)", ['US', 'UK', 'DE', 'FR', 'CA', 'IN'])
    
    st.subheader("구독 및 기기")
    subscription_type = st.selectbox("구독 형태 (Subscription)", ['Free', 'Premium', 'Student'])
    device_type = st.selectbox("주 사용 기기 (Device)", ['Mobile', 'Web', 'Desktop'])

with col2:
    st.subheader("활동 데이터 (일일/주간 평균)")
    listening_time = st.number_input("일일 청취 시간(분)", min_value=0.0, value=60.0, step=5.0)
    songs_played = st.number_input("일일 재생 곡 수", min_value=0, value=20, step=1)
    skip_rate = st.slider("스킵 비율 (Skip Rate)", 0.0, 1.0, 0.3, step=0.01, help="재생한 곡 중 스킵한 비율 (예: 0.3 = 30%)")
    ads_listened = st.number_input("주간 청취 광고 수", min_value=0, value=5, step=1)
    offline_listening = st.radio("오프라인 모드 사용 여부", [0, 1], format_func=lambda x: "사용 안 함" if x==0 else "사용 함")

st.markdown("---")

# ---------------------------------------------------------
# 4. 예측 로직 섹션
# ---------------------------------------------------------
if st.button("🚀 이탈 가능성 예측하기"):
    if pipeline is None:
        st.stop()
        
    # 1) 입력 데이터를 DataFrame으로 변환
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'country': [country],
        'subscription_type': [subscription_type],
        'device_type': [device_type],
        'listening_time': [listening_time],
        'songs_played_per_day': [songs_played],
        'skip_rate': [skip_rate],
        'ads_listened_per_week': [ads_listened],
        'offline_listening': [offline_listening]
    })

    # 2) 🔥 필수: 학습 때와 동일한 Feature Engineering 수행 🔥
    # 앱에서도 이 파생변수들을 만들어줘야 모델이 인식할 수 있습니다.
    input_data['ad_burden'] = input_data['ads_listened_per_week'] / (input_data['listening_time'] + 1)
    input_data['satisfaction_score'] = input_data['songs_played_per_day'] * (1 - input_data['skip_rate'])
    input_data['time_per_song'] = input_data['listening_time'] / (input_data['songs_played_per_day'] + 1)

    # 3) 모델 예측 (파이프라인이 전처리까지 자동으로 수행)
    try:
        # 확률 예측
        churn_proba = pipeline.predict_proba(input_data)[0, 1]
        
        # 임계값 설정 (예: 학습에서 찾은 최적값 0.34 또는 기본값 0.5)
        threshold = 0.35 
        prediction = 1 if churn_proba >= threshold else 0
        
        # ---------------------------------------------------------
        # 5. 결과 출력 섹션
        # ---------------------------------------------------------
        st.header("당신의 유저는...")
        
        # 확률 게이지 표시
        st.metric("이탈 확률 (Churn Probability)", f"{churn_proba*100:.1f}%")
        st.progress(float(churn_proba))

        if prediction == 1:
            st.error("⚠️ **'이탈 위험군 (High Risk)'** 으로 분류되었습니다.")
            st.write("**[액션 플랜 제안]**")
            st.write("- 3개월 할인 쿠폰 푸시 알림 발송")
            st.write("- 최근 많이 스킵한 장르를 제외한 맞춤 플레이리스트 추천")
        else:
            st.success("✅ **'안정적 잔존 유저 (Loyal User)'** 입니다.")
            st.write("현재 활동 패턴이 안정적입니다. 특별한 조치가 필요하지 않습니다.")
            
        # (선택) 디버깅용: 모델이 본 최종 피처 값 확인
        with st.expander("모델 입력 데이터 확인"):
            st.write(input_data)
            
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")
        st.warning("입력 데이터 형식을 확인해주세요.")