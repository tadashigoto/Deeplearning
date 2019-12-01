import numpy as np
import sys
from sklearn import datasets
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
# モデル学習# 数字の手書き画像を取得
mnist = datasets.fetch_mldata('MNIST original',data_home='.')

np.random.seed(123)
n = len(mnist.data)
N = 10000
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

n_in = len(X[0])
n_hidden = 200
n_out = len(Y[0])

model = Sequential()
model.add(Dense(n_hidden,input_dim=n_in))
model.add(Activation('sigmoid'))

model.add(Dense(n_hidden))
model.add(Activation('tanh'))
model.add(Dense(n_hidden))
model.add(Activation('tanh'))
model.add(Dense(n_hidden))
model.add(Activation('tanh'))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01),metrics=['accuracy'])
# plot_model(model, to_file='model.png')
from IPython.display import SVG
from keras.utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
# モデル学習
epochs = 10
batch_size = 100
model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)
# 予測精度の評価
loss_and_metrics = model.evaluate(X_test,Y_test)
print(loss_and_metrics)
