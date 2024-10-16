from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
from RBM import RBM ,train_rbm
#? training RBM trên tập dữ liệu MNIST và save model dưới dạng checkpoint
#> tạo data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
#> Khởi tạo và  RBM
n_visible = 784  # Số lượng pixel trong ảnh MNIST (28x28)
n_hidden = 128    # Số lượng neuron ẩn
rbm = RBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.001)

#% Huấn luyện RBM và lưu giá trị KL Divergence
kl_values = train_rbm(rbm, x_train, epochs=30, batch_size=64, checkpoint_dir='rbm_checkpoint128')

#% Vẽ biểu đồ KL Divergence
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(kl_values) + 1), kl_values, marker='o', linestyle='-')
plt.title('KL Divergence over Epochs')
plt.xlabel('Epochs')
plt.ylabel('KL Divergence')
plt.xticks(np.arange(1, len(kl_values) + 1, step=1))
plt.grid()
plt.show()