# 🎬 MovieLens 1M 기반 AutoInt+ 추천 시스템

본 프로젝트는 MovieLens 1M 데이터를 활용하여 사용자의 취향을 정밀 분석하고, **AutoInt+ (AutoInt + MLP)** 모델을 통해 개인화된 영화를 추천하는 서비스입니다.

## 🚀 주요 수행 작업
1. **AutoInt+ 모델 구현**: Self-Attention과 MLP 레이어를 결합한 하이브리드 아키텍처 설계
2. **기술적 문제 해결**: Keras 3 가중치 로드 오류 수정 및 데이터 타입(int/str) 정합성 확보
3. **Streamlit 서비스 구축**: 사용자 프로필 조회 및 실시간 AI 추천 리스트 시각화

## 📂 폴더 구조
- `autoint.py`: AutoInt+ 모델 아키텍처 정의
- `show_st.py`: Streamlit 웹 서비스 메인 코드
- `fix_weights.py`: 모델 가중치 생성 및 변환 스크립트
- `data/`: 전처리된 MovieLens 데이터셋
- `model/`: 학습된 모델 가중치 (.weights.h5)

## 🏃 실행 방법
1. `python fix_weights.py` (가중치 생성)

2. `streamlit run show_st.py` (서비스 실행)

3. ### 🖼️ 실행 화면
![추천시스템 결과 화면](images/result_screenshot.png)
