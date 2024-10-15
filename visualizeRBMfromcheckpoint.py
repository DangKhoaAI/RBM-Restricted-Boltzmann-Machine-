import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import os
from RBM import RBM  # Nhập lớp RBM từ tệp rbm.py
#>Hàm visualize_reconstruction để trực quan hóa ảnh
def visualize_reconstruction(rbm, data, index=0):
    original_img = data[index].reshape(1, -1)
    reconstructed_img = rbm.reconstruct(original_img)

    original_img = original_img.reshape(28, 28)
    reconstructed_img = reconstructed_img.numpy().reshape(28, 28)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Ảnh gốc
    axes[0].imshow(original_img, cmap="gray")
    axes[0].set_title("Ảnh gốc")
    axes[0].axis('off')

    # Ảnh tái tạo
    axes[1].imshow(reconstructed_img, cmap="gray")
    axes[1].set_title("Ảnh tái tạo")
    axes[1].axis('off')

    plt.show()

#%Tải dữ liệu MNIST
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)

#% Khởi tạo RBM
n_visible = 784  # Số lượng pixel trong ảnh MNIST
n_hidden = 128    # Số lượng neuron ẩn
rbm = RBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.01)

#%Tải mô hình từ checkpoint
rbm.load_model('rbm_checkpoint128')

#% Thực hiện tái tạo và hiển thị ảnh
visualize_reconstruction(rbm, x_train, index=0)
