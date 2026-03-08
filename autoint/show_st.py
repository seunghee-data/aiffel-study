import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from autoint import AutoIntMLP, predict_model

@st.cache_resource
def load_data():
    project_path = os.path.abspath(os.getcwd())
    data_path = os.path.join(project_path, 'data')
    model_path = os.path.join(project_path, 'model')
    
    field_dims = np.load(os.path.join(data_path, 'field_dims.npy'))
    label_encoders = joblib.load(os.path.join(data_path, 'label_encoders.pkl'))
    
    ratings_df = pd.read_csv(os.path.join(data_path, 'ml-1m', 'ratings_prepro.csv'))
    movies_df = pd.read_csv(os.path.join(data_path, 'ml-1m', 'movies_prepro.csv'))
    user_df = pd.read_csv(os.path.join(data_path, 'ml-1m', 'users_prepro.csv'))
    
    model = AutoIntMLP(field_dims, embedding_size=16)
    model(tf.constant([[0] * len(field_dims)], dtype=tf.int32))
    
    weight_file = os.path.join(model_path, 'autoInt_plus.weights.h5')
    if os.path.exists(weight_file):
        model.load_weights(weight_file)
        
    return user_df, movies_df, ratings_df, model, label_encoders

# 데이터 로드
users_df, movies_df, ratings_df, model, label_encoders = load_data()

# --- UI 설정 ---
st.set_page_config(page_title="영화 추천 시스템", layout="wide")
st.title("🎬 AutoInt+ 프리미엄 추천 서비스")

with st.sidebar:
    st.header("👤 사용자 설정")
    user_id = st.number_input("사용자 ID 입력", min_value=1, max_value=int(users_df['user_id'].max()), value=1)
    r_year = st.slider("추천 타겟 연도", 1990, 2010, 2000)
    r_month = st.slider("추천 타겟 월", 1, 12, 1)

if st.button("🚀 분석 및 추천 시작"):
    # 1. 상단 레이아웃 (사용자 정보 & 과거 이력)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 사용자 프로필")
        st.table(users_df[users_df['user_id'] == user_id])

    with col2:
        st.subheader("🍿 과거 선호 이력 (평점 4점 이상)")
        past_ids = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)]['movie_id']
        past_movies = movies_df[movies_df['movie_id'].isin(past_ids)]
        st.dataframe(past_movies[['title', 'genre1', 'movie_year']].head(10))

    st.divider()

    # 2. 추천 엔진 가동 (AutoInt+ 모델)
    st.subheader("🌟 AI 맞춤 추천 리스트 (AutoInt+ 엔진)")
    with st.spinner('사용자의 취향을 정밀 분석 중입니다...'):
        # 안 본 영화 추출
        seen_movies = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].tolist()
        non_seen_movies = movies_df[~movies_df['movie_id'].isin(seen_movies)].copy()
        
        # 유저 정보 복제
        u_info = users_df[users_df['user_id'] == user_id].iloc[0]
        pred_data = non_seen_movies.copy()
        for col in users_df.columns: pred_data[col] = u_info[col]
        
        pred_data['rating_year'], pred_data['rating_month'] = r_year, r_month
        pred_data['rating_decade'] = str(r_year - (r_year % 10)) + 's'
        
        # 피처 정렬 및 인코딩
        cols = ['user_id', 'movie_id', 'movie_decade', 'movie_year', 'rating_year', 
                'rating_month', 'rating_decade', 'genre1', 'genre2', 'genre3', 
                'gender', 'age', 'occupation', 'zip']
        pred_data = pred_data[cols].fillna('no')
        
        for col in cols:
            le = label_encoders[col]
            pred_data[col] = pred_data[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
            pred_data[col] = le.transform(pred_data[col])
        
        # 예측
        recom_results = predict_model(model, pred_data)
        
        # ID 복원 및 영화 정보 매칭
        top_encoded_ids = [r[0] for r in recom_results]
        actual_ids = label_encoders['movie_id'].inverse_transform(top_encoded_ids)
        
        # 최종 결과 추출 (영화 제목, 장르 포함)
        final_movies = movies_df[movies_df['movie_id'].astype(str).isin(actual_ids.astype(str))]
        
        if not final_movies.empty:
            st.dataframe(final_movies[['title', 'genre1', 'genre2', 'movie_year']], use_container_width=True)
        else:
            st.error("추천 결과를 매칭하는 데 실패했습니다.")