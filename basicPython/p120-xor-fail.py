import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD
np.random.seed(0)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(input_dim=2,units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=SGD(lr=0.1))

model.fit(X,Y,epochs=200,batch_size=1)
prob = model.predict_proba(X,batch_size=1)
print(prob)
