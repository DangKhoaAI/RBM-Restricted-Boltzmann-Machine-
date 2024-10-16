import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim  
from RBM_scikitlearn import *  
from tensorflow.keras.datasets import mnist

#> Function to calculate MSE
def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

#> Function to calculate SSIM
def calculate_ssim(original, reconstructed):
    return ssim(original, reconstructed, data_range=1.0)

#> Function to visualize reconstructions
def visualize_reconstruction(rbm, data, num_images=10):
    # Select num_images random indices from 0 to data size
    indices = np.random.choice(data.shape[0], num_images, replace=False)

    # Create figure with 2 rows and num_images columns
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
    
    for i, index in enumerate(indices):
        original_img = data[index].reshape(1, -1)  # Reshape for input
        reconstructed_img = rbm.reconstruct(original_img)

        original_img = original_img.reshape(28, 28)
        reconstructed_img = reconstructed_img.reshape(28, 28)  # Ensure it's in the correct shape

        mse_value = calculate_mse(original_img, reconstructed_img)
        ssim_value = calculate_ssim(original_img, reconstructed_img)

        # Display original image
        axes[0, i].imshow(original_img, cmap="gray")
        axes[0, i].set_title(f"Original Image {i + 1}")
        axes[0, i].axis('off')

        # Display reconstructed image with evaluation metrics
        axes[1, i].imshow(reconstructed_img, cmap="gray")
        axes[1, i].set_title(f"Reconstructed {i + 1}\nMSE: {mse_value:.2f}, SSIM: {ssim_value:.2f}", fontsize=10)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

#% Load MNIST data
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)

#% Initialize RBM
n_visible = 784  # Number of pixels in MNIST image
n_hidden = 256   # Number of hidden neurons
rbm = RBM(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.01)

#% Load model from checkpoint
rbm.load_model('rbm_model.pkl')  

#% Perform reconstruction and display multiple images
visualize_reconstruction(rbm, x_train, num_images=10)
