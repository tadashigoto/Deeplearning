import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
M = 2     # 入力データの次元
K = 3     # クラス数
n = 100   # クラスごとのデータ数
N = n * K # 全データ数
X1 = np.random.randn(n,M)+np.array([0,10])
X2 = np.random.randn(n,M)+np.array([5,5])
X3 = np.random.randn(n,M)+np.array([10,0])
Y1 = np.array([[1,0,0] for i in range(n)])
Y2 = np.array([[0,1,0] for i in range(n)])
Y3 = np.array([[0,0,1] for i in range(n)])
X = np.concatenate((X1,X2,X3), axis=0)
Y = np.concatenate((Y1,Y2,Y3), axis=0)

model = Sequential()
model.add(Dense(input_dim=M,units=K))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1))

# 初期化
minibatch_size = 50
model.fit(X,Y,epochs=20,batch_size=minibatch_size)
n_batches = N
X_,Y_ = shuffle(X,Y)
classes = model.predict_classes(X_[0:10],batch_size=minibatch_size)
prob = model.predict_proba(X_[0:10],batch_size=1)
print('classified:')
print(np.argmax(model.predict(X_[0:10]),axis=1) == classes)
print()
print('output probability:')
print(prob)
