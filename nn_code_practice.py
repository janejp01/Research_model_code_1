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
        self.batchsize = 100
        
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

        # training loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()):
            # for each loop
            for epoch in range(self.epochs):
                for start, end in zip(range(o, len(X), self.batchsize), range(self.batchsize, len(X),self.batchsize)):
                    batch=X[start:end]
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb:prv_hb, _vb:prv_vb})
                    cur_hb =sess.run(update_hb, feed_dict={v0: batch, _w, prv_w,_hb:prv_hb, _vb:prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w, prv_w, _hb:prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb =cur_vb
                error= sess.run(err, feed_dict={v0: X, _w:cur_w, _hb:cur_hb, _vb:cur_vb})
                print('epoch: %d' %epoch, 'reconstructionn error:%f' %error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb =prv_vb
    def rbm_output(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        _vb = tf.constant(self.vb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sees:
            sess.run(tf.global_variables_initializer())
            return see.run(out)

class  NN(object):
    def __init__(self, sizes, X, Y):
        self._sizes = sizes
        self._X = X
        self._Y = Y
        self.w_list = []
        self.b_list = []
        self._learning_rate = 0.1
        self._momentum = 0.0
        self._epoches = 20
        self._batchsize =100
        input_size = X.shape[1]


        # initialization loop
        for size in self._sizes +[Y.shape[1]]:
            #define upper limit for the uniform distribution range
            max_range= 4* math.squt(6. / (input_size +size)) # why need this syntax

            #initialize weights through a random uniform distribution
            self.w_list.append(
                np.random.uniform(-max_range, max_range, [input_size, size]).astype(np.float32)
                )
            # initialize bias as zeroes
            self.b_list.append(np.zeros([size],np.float32))
            input_size =size


    #load data from rbm
    def load_from_rbms(self, dbn_sizes, rbm_list):
        assert len(dbn_sizes) == len(self._sizes)
        for i in range(len(self._sizes)):
            assert dbn_sizes[i] == self._sizes[i]
        for i in range( len(self._sizes)):
            self.w_list[i]= rbm_list[i].w
            self.b_list[i] = rbm_list[i].hb

    def train(self):
        # Create placeholders for input, weights, biases, output
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])

        # define variables and activation function
        for i in range(len(self._sizes)+1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])

        for i in range(1, len(self._sizes)+2):
            _a[i] = tf.nn.sigmoid(tf.matmul(_a[i-1], _w[i-1]) + b[i-1])

        # define the cost function
        cost = tf.reduce_mean(tf.square(_a[-1] -y))

        #define the training operation (Momentum Optimzer minimizing the cost function
        train_op = tf.train.MomentumOptimizer(self._learning_rate, self._momentum).minimize(cost)

        # prediction operation
        predict_op = tf.argmax(_a[-1], 1)

        # train loop
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            #for each epoch
            for i in range(self._epoches):
                # For each step
                for start, end in zip(
                        range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):
                 # Run the training operation on the input data
                    sess.run(train_op, feed_dict={_a[0]:self._X[start:end], y: self._Y[start:end]})


                for j in range (len(self._sizes)+1):
                    # retrieve weights and biases
                    self.w_list[j] = sess.run(_w[j])
                    self.b_list[j] = sess.run(_b[j])

                    
                print("Accuracy rating for epoch " + str(i) + ": " + str(np.mean(np.argmax(self._Y, axis=1) == \
                                                                                 sess.run(predict_op,
                                                                                          feed_dict={_a[0]: self._X,
                                                                                                     y: self._Y}))))





