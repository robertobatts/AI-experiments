# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, PReLU
import matplotlib.pyplot as plt

# load the dataset
dataset = loadtxt('indians-diabetes.csv', delimiter=',')

X = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(16, input_dim=8, kernel_initializer='uniform', name='diabete_model'))
model.add(PReLU())
model.add(Dense(12, kernel_initializer='uniform'))
model.add(PReLU())
model.add(Dense(10, kernel_initializer='uniform'))
model.add(PReLU())
model.add(Dense(6, kernel_initializer='uniform'))
model.add(PReLU())
model.add(Dense(3, kernel_initializer='uniform'))
model.add(PReLU())
model.add(Dense(1, activation='sigmoid'))
  
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, validation_split=0.25, epochs=120, batch_size=5, verbose=1)

model.save('diabete_model')