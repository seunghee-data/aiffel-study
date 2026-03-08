import numpy as np
import tensorflow as tf
import math
from collections import defaultdict
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, BatchNormalization, Embedding
from tensorflow.keras.initializers import TruncatedNormal

class FeaturesEmbedding(Layer):
    def __init__(self, field_dims, embed_dim, **kwargs):
        super(FeaturesEmbedding, self).__init__(**kwargs)
        self.total_dim = sum(field_dims)
        self.embed_dim = embed_dim
        # int32로 고정하여 타입 에러 방지
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        # Keras 3에 맞게 내부 임베딩 레이어를 미리 정의
        self.embedding = Embedding(
            input_dim=self.total_dim, 
            output_dim=self.embed_dim,
            embeddings_initializer='glorot_uniform'
        )

    def call(self, x):
        # 입력값을 int32로 강제 변환 후 오프셋 더하기
        x = tf.cast(x, tf.int32) + tf.constant(self.offsets, dtype=tf.int32)
        return self.embedding(x)

class MultiLayerPerceptron(Layer):
    def __init__(self, input_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, init_std=0.0001, output_layer=True):
        super(MultiLayerPerceptron, self).__init__()
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        hidden_units = [input_dim] + list(hidden_units)
        if output_layer:
            hidden_units += [1]

        self.linears = [Dense(units, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=init_std),
                              kernel_regularizer=tf.keras.regularizers.l2(l2_reg)) for units in hidden_units[1:]]
        self.activation_layer = tf.keras.layers.Activation(activation)
        if self.use_bn:
            self.bn = [BatchNormalization() for _ in hidden_units[1:]]
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        x = inputs
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            if self.use_bn:
                x = self.bn[i](x, training=training)
            x = self.activation_layer(x)
            x = self.dropout(x, training=training)
        return x

class MultiHeadSelfAttention(Layer):
    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, scaling=False, seed=1024, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        self.scaling = scaling

    def build(self, input_shape):
        embedding_size = int(input_shape[-1])
        self.W_Query = self.add_weight(name='query', shape=[embedding_size, self.att_embedding_size * self.head_num], initializer=TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name='key', shape=[embedding_size, self.att_embedding_size * self.head_num], initializer=TruncatedNormal(seed=self.seed + 1))
        self.W_Value = self.add_weight(name='value', shape=[embedding_size, self.att_embedding_size * self.head_num], initializer=TruncatedNormal(seed=self.seed + 2))
        if self.use_res:
            self.W_Res = self.add_weight(name='res', shape=[embedding_size, self.att_embedding_size * self.head_num], initializer=TruncatedNormal(seed=self.seed))
        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        querys = tf.tensordot(inputs, self.W_Query, axes=(-1, 0))
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_Value, axes=(-1, 0))

        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))

        inner_product = tf.matmul(querys, keys, transpose_b=True)
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        normalized_att_scores = tf.nn.softmax(inner_product)

        result = tf.matmul(normalized_att_scores, values)
        result = tf.concat(tf.split(result, self.head_num), axis=-1)
        result = tf.squeeze(result, axis=0) 

        if self.use_res:
            result += tf.tensordot(inputs, self.W_Res, axes=(-1, 0))
        return tf.nn.relu(result)

class AutoInt(Layer):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, init_std=0.0001):
        super(AutoInt, self).__init__()
        self.embedding_layer = FeaturesEmbedding(field_dims, embedding_size)
        self.final_layer = Dense(1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=init_std))
        self.int_layers = [MultiHeadSelfAttention(att_embedding_size=embedding_size, head_num=att_head_num, use_res=att_res) for _ in range(att_layer_num)]

    def call(self, inputs):
        att_input = self.embedding_layer(inputs)
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = Flatten()(att_input)
        att_output = self.final_layer(att_output)
        return tf.nn.sigmoid(att_output)

class AutoIntModel(Model):
    def __init__(self, field_dims, embedding_size, att_layer_num=3, att_head_num=2, att_res=True, dnn_dropout=0, init_std=0.0001):
        super(AutoIntModel, self).__init__()
        self.autoInt_layer = AutoInt(field_dims, embedding_size, att_layer_num=att_layer_num, att_head_num=att_head_num, att_res=att_res, init_std=init_std)

    def call(self, inputs, training=False):
        return self.autoInt_layer(inputs)
    
def predict_model(model, pred_df):
    batch_size = 2048
    top = 10
    user_pred_info = []
    features = pred_df.values
    y_pred = model.predict(features, batch_size=batch_size, verbose=False)
    for feature, p in zip(features, y_pred):
        user_pred_info.append((int(feature[1]), float(p[0])))
    return sorted(user_pred_info, key=lambda s: s[1], reverse=True)[:top]