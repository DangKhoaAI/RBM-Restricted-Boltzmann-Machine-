import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import os
from RBMcustomkeras import RBM  
#> định nghĩa hàm visualize model
def visualize_reconstruction(model, data, index=0):
    original_img = data[index].reshape(1, -1)
    reconstructed_img = model.predict(original_img)

    original_img = original_img.reshape(28, 28)
    reconstructed_img = reconstructed_img.reshape(28, 28)

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
x_test = x_test.reshape(x_test.shape[0], -1)


model_reloaded = tf.keras.models.load_model('model_rbm.h5', custom_objects={'RBM': RBM})
print("Model reloaded from  model_rbm128.h5")
visualize_reconstruction(model_reloaded, x_test, index=0)