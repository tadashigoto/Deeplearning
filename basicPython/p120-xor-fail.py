import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
np.random.seed(0)
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
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(input_dim=2,units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=SGD(lr=0.1))

model.fit(X,Y,epochs=200,batch_size=1)
prob = model.predict_proba(X_[0:10],batch_size=1)
print(prob)
