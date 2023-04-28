from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from util import *
from dnn_np import *
import numpy as np
import matplotlib.pyplot as plt

cfg = Config(num_epoch=100, batch_size=100, learning_rate=0.0005, reg=0.00015)

train_x, train_y, test_x, test_y = get_bat_data()

train_x, _, test_x = normalize(train_x, train_x, test_x)
num_class = (np.unique(train_y)).shape[0]


train_y = train_y.flatten()
test_y = test_y.flatten()
num_class = (np.unique(train_y)).shape[0]
train_y = create_one_hot(train_y, num_class)
test_y = create_one_hot(test_y, num_class)

train_x = add_one(train_x)
test_x = add_one(test_x)

model = Sequential()
model.add(Dense(100, kernel_regularizer= l2(cfg.reg), input_shape = (train_x.shape[1], )))
model.add(Activation('relu'))
model.add(Dense(100, kernel_regularizer= l2(cfg.reg)))
model.add(Activation('relu'))
model.add(Dense(100, kernel_regularizer= l2(cfg.reg)))
model.add(Activation('relu'))
model.add(Dense(num_class))
model.add(Activation('softmax'))
model.summary()

model.compile(optimizer='sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(train_x, train_y, batch_size = cfg.batch_size, epochs = cfg.num_epoch)

model.evaluate(test_x, test_y)

plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'])
plt.show()