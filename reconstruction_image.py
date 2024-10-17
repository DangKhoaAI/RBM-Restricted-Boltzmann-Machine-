# model_loader.py
import tensorflow as tf
import numpy as np
from RBM import RBM  # Replace with the correct import for your RBM class
from RBM_sklearn.RBM_scikitlearn import RBMsklearn  # Replace with your sklearn RBM import
from DeepBeliefNetwork.DBN_deepbeliefnetwork import RBM_dbn  # Replace with your DBN import
from visualize_image import visualize_reconstruction
from tensorflow.keras.datasets import mnist
def load_mnist_data():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1)
    return x_test
def load_model(model_type):
    if model_type == 'dbn':
        model = tf.keras.models.load_model('DeepBeliefNetwork\\model_dbn.h5', custom_objects={'RBM': RBM_dbn})
    elif model_type == 'rbm_sklearn':
        model = RBMsklearn(n_visible=784, n_hidden=256, learning_rate=0.01)
        model.load_model('RBM_sklearn\\rbm_model.pkl')
    elif model_type == 'rbm_tensorflow':
        model = RBM(n_visible=784, n_hidden=128, learning_rate=0.01)
        model.load_model('rbm_checkpoint128')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model
def reconstruct_image(model, img, model_type):
    if model_type == 'dbn':
        rbm_layers = [layer for layer in model.layers if isinstance(layer, RBM_dbn)]
        if len(rbm_layers) != 2:
            raise ValueError("The model must have exactly 2 RBM layers.")
        rbm1, rbm2 = rbm_layers
        hidden_probs1, hidden_sample1 = rbm1.forward(img)
        hidden_probs2, hidden_sample2 = rbm2.forward(hidden_sample1)
        reconstructed_hidden_probs2, reconstructed_hidden_sample2 = rbm2.backward(hidden_sample2)
        reconstructed_img_probs, reconstructed_img_sample = rbm1.backward(reconstructed_hidden_sample2)
        return reconstructed_img_probs.numpy()
    elif model_type =='rbm_sklearn':
        return model.reconstruct(img)
    elif model_type=='rbm_tensorflow':
        return model.reconstruct(img).numpy()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    #% Load MNIST data
    x_test = load_mnist_data()
    #%load moded
    model_type = 'dbn'  # Options: 'dbn', 'rbm_sklearn', 'rbm_tensorflow'
    model = load_model(model_type)
    #% Perform reconstruction and visualize images
    visualize = visualize_reconstruction(num_images=8)
    visualize(reconstruct_image)(model, x_test,model_type)
    