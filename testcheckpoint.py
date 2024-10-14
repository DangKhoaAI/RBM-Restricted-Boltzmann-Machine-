from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import os
from RBM import RBM 
#> Tải mô hình từ checkpoint
(x_train, _), (x_test, _) = mnist.load_data()
x_test = (x_test > 127).astype(np.float32)  # Chuẩn hóa về nhị phân (0, 1)
x_test = x_test.reshape(x_test.shape[0], -1)  # Chuyển đổi thành dạng 1D (28x28 -> 784)

# Khởi tạo RBM với các tham số đã sử dụng trước đó
n_visible = 784  # 28x28 pixels
n_hidden = 64    # Số lượng neuron ẩn
rbm = RBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.01)
rbm.load_model('rbm_checkpoint')

# Kiểm tra xem mô hình có tải thành công không
if rbm.W is not None:
    print("Checkpoint loaded successfully. Here are some of the weights:")
    print(rbm.W.numpy()[:5, :5])  # In một phần nhỏ các trọng số để kiểm tra
else:
    print("Failed to load the model from checkpoint.")
