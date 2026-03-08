import numpy as np
import tensorflow as tf
import os
import pandas as pd
# 우리가 만든 신형 모델을 가져옵니다.
from autoint import AutoIntMLP

# 1. 데이터 경로 설정
project_path = os.path.abspath(os.getcwd())
data_path = os.path.join(project_path, 'data')
movielens_path = os.path.join(data_path, 'ml-1m')

# 2. 필수 파일 로드
field_dims = np.load(os.path.join(data_path, 'field_dims.npy'))
# 학습용 데이터 (이미 전처리가 되어있어야 합니다)
train_df = pd.read_csv(os.path.join(movielens_path, 'ratings_prepro.csv'))

# 3. 모델 초기화
model = AutoIntMLP(field_dims, embedding_size=16)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 딱 1번만 학습 (신형 가중치 파일을 굽는 과정입니다)
# 데이터가 너무 많으면 시간이 걸리니 앞부분 1000개만 맛보기로 학습시켜서 파일만 뽑습니다.
X_train = train_df[['user_id', 'movie_id', 'movie_decade', 'movie_year', 'rating_year', 
                    'rating_month', 'rating_decade', 'genre1', 'genre2', 'genre3', 
                    'gender', 'age', 'occupation', 'zip']].values[:1000]
y_train = train_df['rating'].values[:1000]
y_train = (y_train >= 4).astype(int) # 4점 이상은 1, 아니면 0

print("🚀 신형 AutoInt+ 가중치 생성 중...")
model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)

# 5. 신형 전용 가중치 저장!
model.save_weights(os.path.join(project_path, 'model', 'autoInt_plus_model_weights.h5'))
print("✅ 성공! 'autoInt_plus_model_weights.h5' 파일이 생성되었습니다.")