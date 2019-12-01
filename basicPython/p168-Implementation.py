import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def inference(x, keep_prob, n_in, n_hidden, n_out):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)
    def vias_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)
    # 入力層 - 隠れ層、隠れ層 - 隠れ層
    for i,n_hidden in enumerate(n_hiddens):
        if i == 0:
            input = xinput_dim = n_in
        else:
            input = output
            input_dim = n_hiddens[i-1]
        W = weight_variable([input_dim, n_hidden])
        b = bias_variable([n_hidden])
        h = tf.nn.relu(tf.matmul(input, W) + b)
        output = tf.nn.dropout(h, keep_prob)
    # 隠れ層 - 出力層
    W_out = weight_variable([n_hiddens[-1], n_out])    
    b_out = bias_variable([n_out])
    y = tf.nn.softmax(tf.matmul(output, W_out) + b_out)
    return y

def loss(y, t):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))
    return cross_entropy

def rraining(loss):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_step = optimizer.minimize(loss)
    return train_step

if __name__ == '__main__':
    # 1. データの準備
    # 2. モデルの設定
    n_in = 784
    n_hiddens =[200,200,200] # 各隠れ層の次元数
    n_out = 10
    x = tf.placeholder(tf.float32, shape=[None,n_in])
    keep_prog = tf.placeholder(tf.float32)
    y = inference(x, keep_prog, n_in=n_in n_hiddens=n_hiddens, n_out=n_out)
    # モデル学習
    # モデル評価
