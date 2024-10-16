from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
import numpy as np
from tensorflow.keras.datasets import mnist
import os
#? class RBM use scikitlearn + training and save
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, batch_size=10, n_iter=10):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter

        # Khởi tạo RBM
        self.rbm = BernoulliRBM(n_components=self.n_hidden, learning_rate=self.learning_rate, 
                                batch_size=self.batch_size, n_iter=self.n_iter, verbose=1)
    #>> học rbm
    def contrastive_divergence(self, data):
        # Fit dữ liệu vào RBM
        self.rbm.fit(data)
    #>> tái tạo ảnh ban đầu
    def reconstruct(self, visible):
        # Transform visible units to hidden activations
        hidden_activations = self.rbm.transform(visible)

        # Calculate visible probabilities from hidden activations
        # Get the weight matrix (components_) from the RBM
        weights = self.rbm.components_

        # Compute visible probabilities
        visible_probs = np.dot(hidden_activations, weights)  # Shape: (n_samples, n_visible)

        # Apply sigmoid to get probabilities for each visible unit
        reconstructed_visible = 1 / (1 + np.exp(-visible_probs))

        return reconstructed_visible
    #>> tính kl divergence
    def calculate_kl_divergence(self, data):
        # Reconstruct the visible layer after training
        reconstructed_data = self.reconstruct(data)
        # Calculate KL Divergence
        kl_div = kl_divergence(data, reconstructed_data)
        avg_kl = np.mean(kl_div)
        print(f"KL Divergence after training: {avg_kl:.4f}")
        return avg_kl

    def save_model(self, filepath='rbm_model.pkl'):
        """Lưu mô hình đã học"""
        dump(self.rbm, filepath)

    def load_model(self, filepath='rbm_model.pkl'):
        """Tải mô hình đã lưu"""
        self.rbm = load(filepath)
        print(f"Model loaded from {filepath}")

#> KL Divergence calculation
def kl_divergence(p, q):
    p = np.clip(p, 1e-10, 1.0)  # Avoid log(0)
    q = np.clip(q, 1e-10, 1.0)  # Avoid log(0)
    return np.sum(p * np.log(p / q), axis=1)

#> Load the MNIST dataset
def load_mnist_data():
    # Tải dữ liệu MNIST từ TensorFlow
    (x_train, _), (x_test, _) = mnist.load_data()

    # Reshape từ 28x28 thành vector 784
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Normalize dữ liệu về khoảng [0, 1]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  # Chỉ cần transform cho x_test

    return x_train, x_test

#% Main function to train the RBM on MNIST
if __name__ == "__main__":
    #% Load and preprocess the data
    x_train, _ = load_mnist_data()
    
    #% File path for the saved RBM model
    model_file = "rbm_model.pkl"
    
    #% Check if the model file already exists
    if not os.path.exists(model_file):
        #% Initialize RBM with 784 visible units and 256 hidden units
        rbm = RBM(n_visible=784, n_hidden=256, learning_rate=0.01, batch_size=64, n_iter=10)
        
        #% Train the RBM (this runs for 10 iterations automatically)
        rbm.contrastive_divergence(x_train)
        
        #% Calculate and print KL Divergence after training
        rbm.calculate_kl_divergence(x_train)
        
        #% Save the model
        rbm.save_model()
    else:
        print(f"Model file '{model_file}' already exists. Skipping training and saving.")


    
