import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from skimage.metrics import structural_similarity as ssim
from DBN_deepbeliefnetwork import RBM

def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def calculate_ssim(original, reconstructed):
    return ssim(original, reconstructed, data_range=reconstructed.max() - reconstructed.min())

def reconstruct_image_through_rbm(model, img):
    rbm_layers = [layer for layer in model.layers if isinstance(layer, RBM)]
    
    if len(rbm_layers) != 2:
        raise ValueError("Mô hình cần có đúng 2 lớp RBM.")

    rbm1, rbm2 = rbm_layers
    hidden_probs1, hidden_sample1 = rbm1.forward(img)
    hidden_probs2, hidden_sample2 = rbm2.forward(hidden_sample1)
    
    reconstructed_hidden_probs2, reconstructed_hidden_sample2 = rbm2.backward(hidden_sample2)
    reconstructed_img_probs, reconstructed_img_sample = rbm1.backward(reconstructed_hidden_sample2)
    
    return reconstructed_img_probs

def visualize_reconstruction(model, data, num_images=10):
    indices = np.random.choice(data.shape[0], num_images, replace=False)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i, index in enumerate(indices):
        original_img = data[index].reshape(1, -1)
        reconstructed_img = reconstruct_image_through_rbm(model, original_img)
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

# Tải dữ liệu MNIST
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Nạp mô hình đã lưu
model_reloaded_normal = tf.keras.models.load_model('model_dbn.h5', custom_objects={'RBM': RBM})
print("Model reloaded from model_dbn.h5")

# Hiển thị 8 ảnh ngẫu nhiên từ dữ liệu test
visualize_reconstruction(model_reloaded_normal, x_test, num_images=8)

# Nạp mô hình đã lưu
model_reloaded_transfer = tf.keras.models.load_model('reverted_dbn.h5', custom_objects={'RBM': RBM})
print("Model reloaded from reverted_dbn.h5")

# Hiển thị 8 ảnh ngẫu nhiên từ dữ liệu test
visualize_reconstruction(model_reloaded_transfer, x_test, num_images=8)
