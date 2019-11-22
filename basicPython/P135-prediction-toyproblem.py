import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn import datasets
N = 300
X,y = datasets.make_moons(N,noise=0.3)
from sklearn.model_selection import train_test_split
Y = y.reshape(N, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
num_hidden = 2
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[none, 1])
# 入力値 - 隠れ層
W = tf.Variable(tf.truncated_normal([2,num_hidden]))
b = tf.Variable(tf.zeros([num_hidden]))
h = tf.nn.sigmoid(tf.matmul(x,W)+b)
# 隠れ層 - 出力層
V = tf.valiable(tf.truncated_normal([num_hidden]))
c = tf.Variable(tf.zeros([1]))
y = tf.nn.sigmoid(if.matnul(h,V)+c)

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(rf.greater(y,0.5)),t)
accurace = tf.reduce_men(tf.cast(correct_prediction,tf.float32))

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