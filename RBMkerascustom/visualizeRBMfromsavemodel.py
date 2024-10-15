import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import os
from RBMcustomkeras import RBM
#? visualize reconstruct image qua load RBM model đã train(RBMmodel.h5)
# >Định nghĩa hàm visualize model cho n ảnh ngẫu nhiên
def visualize_reconstruction(model, data, num_images=10):
    # Chọn num_images chỉ số ngẫu nhiên từ 0 đến kích thước của data
    indices = np.random.choice(data.shape[0], num_images, replace=False)
    
    # Tạo figure với 2 hàng và num_images cột
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i, index in enumerate(indices):
        original_img = data[index].reshape(1, -1)
        reconstructed_img = model.predict(original_img)

        original_img = original_img.reshape(28, 28)
        reconstructed_img = reconstructed_img.reshape(28, 28)

        # Ảnh gốc
        axes[0, i].imshow(original_img, cmap="gray")
        axes[0, i].set_title(f"Ảnh gốc {i + 1}")
        axes[0, i].axis('off')

        # Ảnh tái tạo
        axes[1, i].imshow(reconstructed_img, cmap="gray")
        axes[1, i].set_title(f"Ảnh tái tạo {i + 1}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

# Tải dữ liệu MNIST
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Nạp mô hình đã lưu
model_reloaded = tf.keras.models.load_model('RBMmodel.h5', custom_objects={'RBM': RBM})
print("Model reloaded from RBMmodel.h5")

# Hiển thị 10 ảnh ngẫu nhiên
visualize_reconstruction(model_reloaded, x_test, num_images=8)
