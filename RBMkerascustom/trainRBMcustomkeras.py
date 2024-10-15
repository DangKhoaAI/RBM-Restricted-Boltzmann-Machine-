import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from RBMcustomkeras import *
import os
#? file này train RBM trên dữ liệu mnist và save weight model
# Dữ liệu training
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)

#%Tạo kiến trúc model
n_hidden = 128 # Số lượng neuron ẩn
rbm = RBM(n_hidden=n_hidden, learning_rate=0.01)
inputs = Input(shape=(784,))
outputs=rbm(inputs)
model = tf.keras.Model(inputs,outputs)
model.build((None,784))
#%train model
train_rbm(model, x_train, epochs=20)
model.save('model_rbm.h5')
print("Save model at model_rbm.h5")


