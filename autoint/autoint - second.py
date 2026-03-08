import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, BatchNormalization, Embedding, Activation
from tensorflow.keras.initializers import TruncatedNormal

# 1. 임베딩 레이어
class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        self.embedding = Embedding(input_dim=sum(field_dims), output_dim=embed_dim)

    def call(self, x):
        x = tf.cast(x, tf.int32) + tf.constant(self.offsets, dtype=tf.int32)
        return self.embedding(x)

# 2. MLP 레이어
class MultiLayerPerceptron(Layer):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.4, init_std=0.0001, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = [Dense(units, kernel_initializer=tf.random_normal_initializer(stddev=init_std)) for units in hidden_units]
        self.output_layer = Dense(1) if output_layer else None
        self.dropout = Dropout(dropout_rate)
        self.activation = Activation('relu')

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        if self.output_layer:
            x = self.output_layer(x)
        return x

# 3. Attention 레이어
class MultiHeadSelfAttention(Layer):
    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.att_embedding_size, self.head_num, self.use_res = att_embedding_size, head_num, use_res

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.W_Q = self.add_weight(shape=[dim, self.att_embedding_size * self.head_num])
        self.W_K = self.add_weight(shape=[dim, self.att_embedding_size * self.head_num])
        self.W_V = self.add_weight(shape=[dim, self.att_embedding_size * self.head_num])
        if self.use_res: self.W_R = self.add_weight(shape=[dim, self.att_embedding_size * self.head_num])

    def call(self, inputs):
        q = tf.stack(tf.split(tf.tensordot(inputs, self.W_Q, axes=(-1, 0)), self.head_num, axis=2))
        k = tf.stack(tf.split(tf.tensordot(inputs, self.W_K, axes=(-1, 0)), self.head_num, axis=2))
        v = tf.stack(tf.split(tf.tensordot(inputs, self.W_V, axes=(-1, 0)), self.head_num, axis=2))
        score = tf.nn.softmax(tf.matmul(q, k, transpose_b=True))
        out = tf.concat(tf.split(tf.matmul(score, v), self.head_num), axis=-1)
        out = tf.squeeze(out, axis=0)
        if self.use_res: out += tf.tensordot(inputs, self.W_R, axes=(-1, 0))
        return tf.nn.relu(out)

# 4. AutoInt+ (AutoIntMLP) 최종 모델
class AutoIntMLP(Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, dnn_hidden_units=(32, 32)):
        super(AutoIntMLP, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embedding_size)
        self.int_layers = [MultiHeadSelfAttention(embedding_size, att_head_num) for _ in range(att_layer_num)]
        self.dnn_linear = Dense(1)
        self.dnn = MultiLayerPerceptron(len(field_dims) * embedding_size, dnn_hidden_units)

    def call(self, inputs, training=False):
        embed_x = self.embedding(inputs)
        att_input = embed_x
        for layer in self.int_layers: att_input = layer(att_input)
        att_out = self.dnn_linear(Flatten()(att_input))
        dnn_out = self.dnn(Flatten()(embed_x), training=training)
        return tf.nn.sigmoid(att_out + dnn_out)

# 추천 예측 함수
def predict_model(model, pred_df):
    y_pred = model.predict(pred_df.values, batch_size=2048, verbose=False)
    user_pred_info = [(int(f[1]), float(p[0])) for f, p in zip(pred_df.values, y_pred)]
    return sorted(user_pred_info, key=lambda s: s[1], reverse=True)[:10]