# visualizeimage.py
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
#? thiết kế mse ,ssim và làm decorator visualize reconstruction image (đưa qua reconstruction_image.py)
def calculate_mse(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def calculate_ssim(original, reconstructed):
    return ssim(original, reconstructed, data_range=1.0)

def visualize_reconstruction(num_images=10):
    def decorator(reconstruct_fn):
        def wrapper(model, data,modeltype):
            indices = np.random.choice(data.shape[0], num_images, replace=False)
            fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))

            for i, index in enumerate(indices):
                original_img = data[index].reshape(1, -1)
                reconstructed_img = reconstruct_fn(model, original_img,modeltype)

                original_img = original_img.reshape(28, 28)
                reconstructed_img = reconstructed_img.reshape(28, 28)

                mse_value = calculate_mse(original_img, reconstructed_img)
                ssim_value = calculate_ssim(original_img, reconstructed_img)

                # Display original image
                axes[0, i].imshow(original_img, cmap="gray")
                axes[0, i].set_title(f"Original {i + 1}")
                axes[0, i].axis('off')

                # Display reconstructed image
                axes[1, i].imshow(reconstructed_img, cmap="gray")
                axes[1, i].set_title(f"Reconstructed {i + 1}\nMSE: {mse_value:.2f}, SSIM: {ssim_value:.2f}", fontsize=10)
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.show()

        return wrapper
    return decorator
