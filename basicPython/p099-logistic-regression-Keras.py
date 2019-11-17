# 3.4.3.2 ロジスティック回帰Kerasによる実装 (P.99)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
model = Sequential([
    Dense(imput_dim=2, units=-1),
    Activation('sigmoid')
])
# 確率的勾配降下法
model.compile(loss='binary_crossentropy',optimizer=SGD(lr=0.1)) # lr(learning late)学習率
'''
モデル学習
'''
# ORゲート
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[1]])

model.fit(X,Y, epochs=200,batch_size=1)
# 学習結果の確認
classes = model.predit_classes(X,batch_size=1)
prob = model.predict_proba(X,Batch_size=1)

print('classified:')
print(Y == classes)
print()
print('output probalilitey:')
print(prob)
