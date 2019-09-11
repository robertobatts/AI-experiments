# first neural network with keras make predictions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
dataset = loadtxt('indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
testX = dataset[0:100:,0:8]
testy = dataset[0:100:,8]
X = dataset[100:,0:8]
y = dataset[100:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
#evaluate the keras model
loss, accuracy = model.evaluate(testX, testy)
print('Accuracy: %.2f, Loss: %.2f' % (accuracy*100, loss*100) )

# make class predictions with the model
#predictions = model.predict_classes(testX)

#print('%d tests passed' % passed)