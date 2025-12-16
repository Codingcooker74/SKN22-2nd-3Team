import streamlit as st

st.set_page_config(page_title="이탈 예측 확인", page_icon="🔍", layout="wide")

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    font-family: 'Segoe UI', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.title("🔍 이탈 예측 확인 페이지")
st.markdown("여기에서 고객 이탈 예측 결과를 확인하거나 모델을 테스트할 수 있습니다.")