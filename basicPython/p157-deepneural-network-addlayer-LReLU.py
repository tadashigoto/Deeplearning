'''
Leaky ReLU
'''
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn import datasets
from sklearn.utils import shuffle
def lrelu(x,alpha=0.0):
    return tf.maximum(alpha*x,x)
np.random.seed(0)
tf.set_random_seed(1234)
N = 300
X,y = datasets.make_moons(N,noise=0.3)
from sklearn.model_selection import train_test_split
Y = y.reshape(N, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

n_in = len(X[0]) # 784
n_hidden = 200   
n_out = len(Y[0]) # 10

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
# 入力値 - 隠れ層
W0 = tf.Variable(tf.truncated_normal([n_in,n_hidden],stddev=0.01))
b0 = tf.Variable(tf.zeros([n_hidden]))
h0 = lrelu(tf.matmul(x,W0)+b0)
# 隠れ値 - 隠れ層
W1 = tf.Variable(tf.truncated_normal([n_hidden,n_hidden],stddev=0.01))
b1 = tf.Variable(tf.zeros([n_hidden]))
h1 = lrelu(tf.matmul(h0,W1)+b1)
# 隠れ値 - 隠れ層
W2 = tf.Variable(tf.truncated_normal([n_hidden,n_hidden],stddev=0.01))
b2 = tf.Variable(tf.zeros([n_hidden]))
h2 = lrelu(tf.matmul(h1,W2)+b2)
# 隠れ値 - 隠れ層
W3 = tf.Variable(tf.truncated_normal([n_hidden,n_hidden],stddev=0.01))
b3 = tf.Variable(tf.zeros([n_hidden]))
h3 = lrelu(tf.matmul(h2,W3)+b3)
# 隠れ層 - 出力層
W4 = tf.Variable(tf.truncated_normal([n_hidden,n_out],stddev=0.01))
b4 = tf.Variable(tf.zeros([n_out]))
y = tf.nn.sigmoid(tf.matmul(h3,W4)+b4)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y,0.5)),t)
accurace = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

batch_size = 20
n_batches = N // batch_size

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(500):
    X_, Y_= shuffle(X_train,Y_train)
    for i in range(n_batches):
        start = i * batch_size
        end = start * batch_size
        sess.run(train_step, feed_dict ={
            x: X_[start:end],
            t: Y_[start:end]
        })
accurace_rate = accurace.eval(session=sess,feed_dict = {
    x:X_test,
    t:Y_test
})
print('accuracy:',accurace_rate)
