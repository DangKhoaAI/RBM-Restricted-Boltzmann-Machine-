import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

#> Load the trained model
supervised_model = tf.keras.models.load_model('SupervisedwpreRBM.h5')

#> Load the MNIST data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype(np.float32) / 255.0
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

#> Convert labels to one-hot encoding for evaluation
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

#> Make predictions
predictions = supervised_model.predict(x_test_flattened)
predicted_labels = np.argmax(predictions, axis=1)

#> Function to visualize the image with true and predicted labels
def visualize_predictions(images, true_labels, predicted_labels, num_samples=10):
    # Randomly select indices
    random_indices = np.random.choice(images.shape[0], num_samples, replace=False)
    
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx], cmap='gray')
        plt.title(f"True: {true_labels[idx]}\nPred: {predicted_labels[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


#%Visualize 10 samples from the test set with true and predicted labels
visualize_predictions(x_test, y_test, predicted_labels, num_samples=10)
