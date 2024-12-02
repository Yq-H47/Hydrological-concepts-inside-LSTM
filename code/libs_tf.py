import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()


class LSTM(Layer):
    def __init__(self, input_xd, hidden_size, seed=200,**kwargs):
        self.input_xd = input_xd
        self.hidden_size = hidden_size
        self.seed = seed
        super(LSTM, self).__init__(**kwargs)

    def build(self, input_shape):

        self.w_ih = self.add_weight(name='w_ih', shape=(self.input_xd, 4 * self.hidden_size),
                                 initializer=initializers.Orthogonal(seed=self.seed - 5),
                                 trainable=True)

        self.w_hh = self.add_weight(name='w_hh',
                                       shape=(self.hidden_size, 4 * self.hidden_size),
                                       initializer=initializers.Orthogonal(seed=self.seed + 5),
                                       trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(4 * self.hidden_size, ),
                                    #initializer = 'random_normal',
                                    initializer=initializers.Constant(value=0),
                                    trainable=True)

        self.shape = input_shape
        self.reset_parameters()
        super(LSTM, self).build(input_shape)


    def reset_parameters(self):
        #self.w_ih.initializer = initializers.Orthogonal(seed=self.seed - 5)
        #self.w_sh.initializer = initializers.Orthogonal(seed=self.seed + 5)

        w_hh_data = K.eye(self.hidden_size)
        #bias_s_batch = K.repeat_elements(bias_s_batch, rep=sample_size_d, axis=0)
        w_hh_data = K.repeat_elements(w_hh_data, rep=4, axis=1)
        self.w_hh = w_hh_data

        #self.bias.initializer = initializers.Constant(value=0)
        #self.bias_s.initializer = initializers.Constant(value=0)

    def call(self, inputs_x):
        forcing = inputs_x  #[batch, seq_len, dim]
        print('forcing_shape:',forcing.shape)
        #attrs = inputs_x[1]     #[batch, dim]
        #print('attrs_shape:',attrs.shape)

        forcing_seqfir = K.permute_dimensions(forcing, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        print('forcing_seqfir_shape:',forcing_seqfir.shape)

        #attrs_seqfir = K.permute_dimensions(attrs, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        #print('attrs_seqfir_shape:',attrs_seqfir.shape)


        seq_len = forcing_seqfir.shape[0]
        print('seq_len:',seq_len)
        batch_size = forcing_seqfir.shape[1]
        print('batch_size:',batch_size)

        #init_states = [K.zeros((K.shape(forcing)[0], 2))]
        #h0, c0 = [K.zeros(shape= (sample_size_d,self.hidden_size)),K.zeros(shape= (sample_size_d,self.hidden_size))]
        h0 = K.zeros(shape= (batch_size, self.hidden_size))
        c0 = K.zeros(shape= (batch_size, self.hidden_size))
        h_x = (h0, c0)

        h_n, c_n = [], []

        bias_batch = K.expand_dims(self.bias, axis=0)
        bias_batch = K.repeat_elements(bias_batch, rep=batch_size, axis=0)
        print("bias_batch:",bias_batch.shape)

        #bias_s_batch = K.expand_dims(self.bias_s, axis=0)
        #bias_s_batch = K.repeat_elements(bias_s_batch, rep=batch_size, axis=0)
        #i = K.sigmoid(K.dot(attrs, self.w_sh) + bias_s_batch)

        for t in range(seq_len):
            h_0, c_0 = h_x
            gates =((K.dot(h_0, self.w_hh) + bias_batch) + K.dot(forcing_seqfir[t], self.w_ih))
            f, i, o, g = tf.split(value=gates, num_or_size_splits=4, axis=1)

            next_c = K.sigmoid(f) * c_0 + K.sigmoid(i) * K.tanh(g)
            next_h = K.sigmoid(o) * K.tanh(next_c)


            h_n.append(next_h)
            c_n.append(next_c)

            h_x = (next_h,next_c)

        h_n = K.stack(h_n, axis=0)
        c_n = K.stack(c_n, axis=0)

        c_n = K.permute_dimensions(c_n, pattern=(1, 0, 2))

        return c_n

class LSTM_tsetQ(Layer):
    def __init__(self, input_xd, hidden_size, seed=200,**kwargs):
        self.input_xd = input_xd
        self.hidden_size = hidden_size
        self.seed = seed
        super(LSTM_tsetQ, self).__init__(**kwargs)

    def build(self, input_shape):

        self.w_ih = self.add_weight(name='w_ih', shape=(self.input_xd, 4 * self.hidden_size),
                                 initializer=initializers.Orthogonal(seed=self.seed - 5),
                                 trainable=True)

        self.w_hh = self.add_weight(name='w_hh',
                                       shape=(self.hidden_size, 4 * self.hidden_size),
                                       initializer=initializers.Orthogonal(seed=self.seed + 5),
                                       trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=(4 * self.hidden_size, ),
                                    #initializer = 'random_normal',
                                    initializer=initializers.Constant(value=0),
                                    trainable=True)

        self.shape = input_shape
        self.reset_parameters()
        super(LSTM_tsetQ, self).build(input_shape)


    def reset_parameters(self):
        #self.w_ih.initializer = initializers.Orthogonal(seed=self.seed - 5)
        #self.w_sh.initializer = initializers.Orthogonal(seed=self.seed + 5)

        w_hh_data = K.eye(self.hidden_size)
        #bias_s_batch = K.repeat_elements(bias_s_batch, rep=sample_size_d, axis=0)
        w_hh_data = K.repeat_elements(w_hh_data, rep=4, axis=1)
        self.w_hh = w_hh_data

        #self.bias.initializer = initializers.Constant(value=0)
        #self.bias_s.initializer = initializers.Constant(value=0)

    def call(self, inputs_x):
        forcing = inputs_x  #[batch, seq_len, dim]
        print('forcing_shape:',forcing.shape)
        #attrs = inputs_x[1]     #[batch, dim]
        #print('attrs_shape:',attrs.shape)

        forcing_seqfir = K.permute_dimensions(forcing, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        print('forcing_seqfir_shape:',forcing_seqfir.shape)

        #attrs_seqfir = K.permute_dimensions(attrs, pattern=(1, 0, 2))  #[seq_len, batch, dim]
        #print('attrs_seqfir_shape:',attrs_seqfir.shape)


        seq_len = forcing_seqfir.shape[0]
        print('seq_len:',seq_len)
        batch_size = forcing_seqfir.shape[1]
        print('batch_size:',batch_size)

        #init_states = [K.zeros((K.shape(forcing)[0], 2))]
        #h0, c0 = [K.zeros(shape= (sample_size_d,self.hidden_size)),K.zeros(shape= (sample_size_d,self.hidden_size))]
        h0 = K.zeros(shape= (batch_size, self.hidden_size))
        c0 = K.zeros(shape= (batch_size, self.hidden_size))
        h_x = (h0, c0)

        h_n, c_n = [], []

        bias_batch = K.expand_dims(self.bias, axis=0)
        bias_batch = K.repeat_elements(bias_batch, rep=batch_size, axis=0)
        print("bias_batch:",bias_batch.shape)

        for t in range(seq_len):
            h_0, c_0 = h_x

            gates =((K.dot(h_0, self.w_hh) + bias_batch) + K.dot(forcing_seqfir[t], self.w_ih))
            f, i, o, g = tf.split(value=gates, num_or_size_splits=4, axis=1)

            next_c = K.sigmoid(f) * c_0 + K.sigmoid(i) * K.tanh(g)
            next_h = K.sigmoid(o) * K.tanh(next_c)


            h_n.append(next_h)
            c_n.append(next_c)

            h_x = (next_h,next_c)

        h_n = K.stack(h_n, axis=0)
        c_n = K.stack(c_n, axis=0)

        #c_n = K.permute_dimensions(c_n, pattern=(1, 0, 2))

        return c_n,h_n

#
class LinearRegression(Layer):
    def __init__(self, output_dim, seed=200, **kwargs):
        self.output_dim = output_dim
        self.seed = seed
        super(LinearRegression, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[2], self.output_dim),
                                      initializer=initializers.RandomNormal(seed=self.seed),
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer=initializers.Constant(value=0),
                                    trainable=True)
        super(LinearRegression, self).build(input_shape)

    '''def call(self, inputs):
        output = K.dot(inputs, self.kernel) + self.bias
        return output'''
    def call(self, inputs):
        weighted = K.dot(inputs, self.kernel)
        output = weighted + self.bias
        return output


class ScaleLayer(Layer):

    def __init__(self, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, inputs):

        forc1 = inputs[:, :, :5]

        q_mean = K.constant(1.616004426269345, dtype='float32')
        q_std = K.constant(3.782566266062373, dtype='float32')

        forc2 = (inputs[:,:,5:6] - q_mean)/q_std

        #similar = inputs[:, :, 6:7]

        return K.concatenate([forc1,forc2], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

class ScaleLayer_car(Layer):


    def __init__(self, **kwargs):
        super(ScaleLayer_car, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer_car, self).build(input_shape)

    def call(self, inputs):
        #met(气象输入) = [wrap_number_train, wrap_length, 9('prcp(mm/day)', 'tmean(C)', 'dayl(day)', 'srad(W/m2)', 'vp(Pa)'), et q1 q2 m]


        forc1 = inputs[:, :, :5]

        forc2 = inputs[:, :, 5:6]
        self.forc2_center = K.mean(forc2, axis=-2, keepdims=True)
        self.forc2_scale = K.std(forc2, axis=-2, keepdims=True)
        self.forc2_scaled = (forc2 - self.forc2_center) / self.forc2_scale

        similar = inputs[:, :, 6:7]

        intervariable = inputs[:, :, 7:11]
        self.intervariable_center = K.mean(intervariable, axis=-2, keepdims=True)
        self.intervariable_scale = K.std(intervariable, axis=-2, keepdims=True)
        self.intervariablet_scaled = (intervariable - self.intervariable_center) / self.intervariable_scale

        return K.concatenate([forc1,self.forc2_scaled,similar,self.intervariablet_scaled], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape

class ScaleLayer_caravan(Layer):

    def __init__(self, **kwargs):
        super(ScaleLayer_caravan, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer_caravan, self).build(input_shape)

    def call(self, inputs):

        met = inputs[:, :, 0:5]

        attrs = inputs[:, :, 5:-4]

        physic_four = inputs[:, :, -4:]

        self.physic_four_center = K.mean(physic_four, axis=-2, keepdims=True)
        self.physic_four_scale = K.std(physic_four, axis=-2, keepdims=True)
        self.physic_four_scaled = (physic_four - self.physic_four_center) / self.physic_four_scale

        return K.concatenate([met, attrs, self.physic_four_scaled], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape


















