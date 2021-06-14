import math
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class RBM(object):
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        self.epochs = 5 # can add int number by myself
        self.learning_rate = 1.0
        self.batch_size = 100
        
        self.w = np.zeros([input_size, output_size], np.float32)
        self.hb = np.zeros([output_size], np.float32)
        self.vb = np.zeros([input_size], np.float32)
        
    def prob_h_given_v(self, visible, w, hb):
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)
    def sample_prob(self,probs):
        return tf.nn.relu(tf.sign(probs-tf.random_uniform(tf.shape(probs))))
    def train(self, X):
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        
        #creates and initializes the weights wit 0
        prv_w = np.zeros([self._input_size, self._output_size], np.float32)
        #creates and initializes the hidden biases with 0
        prv_hb = np.zeros([self._output_size], np.float32)
        # creates and initializes the visible biases with 0
        prv_vb = np.zeros([self._input_size], np.float32)

        cur_w = np.zeros([self._input_size, self._output_size], np.float32)
        cur_hb = np.zeros([self._output_size], np.float32)
        cur_vb = np.zeros([self._input_size], np.float32)
        v0 = tf.placeholder("float", [None, self._input_size])
        
        # initializes with sample probabilities
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))
        v1 = self.sample_prob(self.prob_h_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        # create the gradients
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.math(tf.transpose(v1), h1)


        # update learning rates for the layers
        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.to_float(tf.shape(v0[0]))
        update_hb = _hb + self.learning_rate* tf.reduce_mean(h0-h1 , 0)
        update_vb= _vb + self.learning_rate * tf.reduce_mean(v0-v1, 0)

        # find the error rate
        err= tf.reduce_mean(tf.square(v0-v1)