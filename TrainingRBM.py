from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
from RBM import RBM ,train_rbm
(x_train, _), (x_test, _) = mnist.load_data()
x_train = (x_train > 127).astype(np.float32)  # Chuẩn hóa về nhị phân (0, 1)
x_train = x_train.reshape(x_train.shape[0], -1)

#> Khởi tạo và huấn luyện RBM
n_visible = 784  # Số lượng pixel trong ảnh MNIST (28x28)
n_hidden = 128    # Số lượng neuron ẩn
rbm = RBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.01)

train_rbm(rbm, x_train,epochs=20, batch_size=64 ,checkpoint_dir='rbm_checkpoint128')

