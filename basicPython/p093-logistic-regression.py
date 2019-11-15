# 3.4 ロジスティック回帰 (P.93)
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(0)

w = tf.Variable(tf.zeros([2,1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape=[None,2])
t = tf.placeholder(tf.float32, shape=[None,1])
y = tf.nn.sigmoid(tf.matmul(x,w)+b) # matmul:xとwの内積

# 交差エントロピー誤差関数
cross_entropy = tf.reduce_sum(t * tf.log(y) + (1-t) * tf.log(1-y))
# 確率的勾配降下法(学習率0.1)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.to_float(tf.greater(y,0.5)),t) # ニューロンの発火

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[1]])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
# 学習
for epoch in range(200):
    sess.run(train_step, feed_dict ={
        x: X,
        t: Y
    })
# 学習結果の確認
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
prob = y.eval(session=sess,feed_dict={
    x: X
})
print('classified')
print(classified)
print()
print('output probalilitey:')
print(prob)
print('w:',sess.run(w))