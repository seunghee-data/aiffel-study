import numpy as np
import tensorflow as tf
import os
from autoint import AutoIntMLP

# 경로 설정
project_path = os.path.abspath(os.getcwd())
model_path = os.path.join(project_path, 'model')
data_path = os.path.join(project_path, 'data')

# 모델 빌드를 위해 데이터 규격 로드
field_dims = np.load(os.path.join(data_path, 'field_dims.npy'))
model = AutoIntMLP(field_dims, embedding_size=16)

# 더미 데이터로 모델 빌드
model(tf.constant([[0] * len(field_dims)], dtype=tf.int32))

# 최신 Keras 규격(.weights.h5)으로 저장
weight_path = os.path.join(model_path, 'autoInt_plus.weights.h5')
model.save_weights(weight_path)
print(f"✅ 새 가중치 파일 생성 완료: {weight_path}")