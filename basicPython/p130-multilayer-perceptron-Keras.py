from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import SGD

# np.random.seed(123)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(input_dim=2,units=2))
model.add(Activation('sigmoid'))
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=SGD(lr=0.1))

# 初期化
model.fit(X,Y,epochs=16000,batch_size=4)
classes = model.predict_classes(X,batch_size=4)
prob = model.predict_proba(X,batch_size=4)
print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)
