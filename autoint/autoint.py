import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, Embedding

# 1. 임베딩 레이어
class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        self.embedding = Embedding(input_dim=sum(field_dims), output_dim=embed_dim)
    def call(self, x):
        x = tf.cast(x, tf.int32) + tf.constant(self.offsets, dtype=tf.int32)
        return self.embedding(x)

# 2. MLP 레이어 (AutoInt+의 핵심)
class MultiLayerPerceptron(Layer):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.4):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = [Dense(units, activation='relu') for units in hidden_units]
        self.output_layer = Dense(1)
        self.dropout = Dropout(dropout_rate)
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x, training=training)
        return self.output_layer(x)

# 3. Attention 레이어
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_size, head_num, use_res=True, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_size, self.head_num, self.use_res = embed_size, head_num, use_res
    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.W_Q = self.add_weight(shape=[dim, self.embed_size * self.head_num], name="W_Q")
        self.W_K = self.add_weight(shape=[dim, self.embed_size * self.head_num], name="W_K")
        self.W_V = self.add_weight(shape=[dim, self.embed_size * self.head_num], name="W_V")
        if self.use_res: self.W_R = self.add_weight(shape=[dim, self.embed_size * self.head_num], name="W_R")
    def call(self, inputs):
        q = tf.stack(tf.split(tf.tensordot(inputs, self.W_Q, axes=(-1, 0)), self.head_num, axis=2))
        k = tf.stack(tf.split(tf.tensordot(inputs, self.W_K, axes=(-1, 0)), self.head_num, axis=2))
        v = tf.stack(tf.split(tf.tensordot(inputs, self.W_V, axes=(-1, 0)), self.head_num, axis=2))
        score = tf.nn.softmax(tf.matmul(q, k, transpose_b=True))
        out = tf.concat(tf.split(tf.matmul(score, v), self.head_num), axis=-1)
        out = tf.squeeze(out, axis=0)
        if self.use_res: out += tf.tensordot(inputs, self.W_R, axes=(-1, 0))
        return tf.nn.relu(out)

# 4. AutoIntMLP 최종 모델 (Joint Training 구조)
class AutoIntMLP(Model): 
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, dnn_hidden_units=(32, 32)):
        super(AutoIntMLP, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)
        self.num_fields = len(field_dims)
        self.embedding_size = embedding_size
        self.att_final = Dense(1, use_bias=False)
        self.dnn = MultiLayerPerceptron(self.num_fields * embedding_size, dnn_hidden_units)
        self.int_layers = [MultiHeadSelfAttention(embedding_size, att_head_num) for _ in range(att_layer_num)]

    def call(self, inputs, training=False):
        embed_x = self.embedding(inputs)
        # Attention 경로
        att_in = embed_x
        for layer in self.int_layers: att_in = layer(att_in)
        att_out = self.att_final(Flatten()(att_in))
        # DNN 경로
        dnn_in = tf.reshape(embed_x, shape=(-1, self.num_fields * self.embedding_size))
        dnn_out = self.dnn(dnn_in, training=training)
        # 결과 합산 (Joint)
        return tf.nn.sigmoid(att_out + dnn_output) if 'dnn_output' in locals() else tf.nn.sigmoid(att_out + dnn_out)

def predict_model(model, pred_df):
    y_pred = model.predict(pred_df.values, batch_size=1024, verbose=False)
    # 인덱스와 점수를 정확히 매칭하여 상위 10개 반환
    res = [(int(f[1]), float(p[0])) for f, p in zip(pred_df.values, y_pred)]
    return sorted(res, key=lambda x: x[1], reverse=True)[:10]