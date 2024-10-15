import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim  # Đảm bảo đã cài đặt scikit-image
from RBM import RBM  # Nhập lớp RBM từ tệp rbm.py
from tensorflow.keras.datasets import mnist
#> Hàm tính toán MSE
def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

#> Hàm tính toán SSIM
def calculate_ssim(original, reconstructed):
    return ssim(original, reconstructed, data_range=1.0)

#> Hàm visualize_reconstruction để trực quan hóa nhiều ảnh
def visualize_reconstruction(rbm, data, num_images=10):
    # Chọn num_images chỉ số ngẫu nhiên từ 0 đến kích thước của data
    indices = np.random.choice(data.shape[0], num_images, replace=False)

    # Tạo figure với 2 hàng và num_images cột
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i, index in enumerate(indices):
        original_img = data[index].reshape(1, -1)
        reconstructed_img = rbm.reconstruct(original_img)

        original_img = original_img.reshape(28, 28)
        reconstructed_img = reconstructed_img.numpy().reshape(28, 28)

        mse_value = calculate_mse(original_img, reconstructed_img)
        ssim_value = calculate_ssim(original_img, reconstructed_img)

        # Hiển thị ảnh gốc
        axes[0, i].imshow(original_img, cmap="gray")
        axes[0, i].set_title(f"Ảnh gốc {i + 1}")
        axes[0, i].axis('off')

        # Hiển thị ảnh tái tạo và đánh giá với cỡ chữ nhỏ hơn
        axes[1, i].imshow(reconstructed_img, cmap="gray")
        axes[1, i].set_title(f"Ảnh tái tạo {i + 1}\nMSE: {mse_value:.2f}, SSIM: {ssim_value:.2f}", fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

#% Tải dữ liệu MNIST
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)

# %Khởi tạo RBM
n_visible = 784  # Số lượng pixel trong ảnh MNIST
n_hidden = 128   # Số lượng neuron ẩn
rbm = RBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.01)

#% Tải mô hình từ checkpoint
rbm.load_model('rbm_checkpoint128')

#% Thực hiện tái tạo và hiển thị nhiều ảnh
visualize_reconstruction(rbm, x_train, num_images=10)
