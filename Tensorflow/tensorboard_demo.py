#%%
from tensorflow.keras import layers,models,datasets,optimizers,losses
from tensorflow.keras import Input
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
import numpy as np
import matplotlib.pyplot as plt

#%%
train, test = datasets.mnist.load_data()

train_x, train_y = train
test_x, test_y = test

train_x, train_y = train_x.astype(np.float32), train_y.astype(np.float32)
test_x, test_y = test_x.astype(np.float32), test_y.astype(np.float32)

train_x, test_x = np.expand_dims(train_x, axis = 3), np.expand_dims(test_x, axis = 3)

print('X shape : ', train_x.shape)
#%%
model = models.Sequential()

model.add(Input(shape = (28,28,1)))

model.add(layers.Conv2D(16,kernel_size=(4,4), strides=(1,1), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(1,1)))

model.add(layers.Conv2D(16,kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2), strides=(1,1)))

model.add(layers.Flatten())

model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

opt = optimizers.Adam(learning_rate = 0.001)
loss_object = losses.SparseCategoricalCrossentropy()

model.compile(optimizer = opt, loss = loss_object, metrics=['accuracy']) 
print(model.summary())

#%%
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
model.fit(x = train_x, y = train_y, batch_size=500, epochs = 5, callbacks=tensorboard)
# %%
