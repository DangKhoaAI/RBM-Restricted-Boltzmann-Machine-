import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from RBMcustomkeras import RBM  # Import the RBM class
import time
#? file so sánh giữa 2 model : 1.supervised thông thường , supervised with pretraing RBM
# >Load the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# >convert labels to one-hot encoding
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

#> Function to create a model
def create_supervised_model(pretrained_weights=None):
    model = Sequential()
    model.add(InputLayer(input_shape=(784,)))
    if pretrained_weights:
        model.add(Dense(128, activation='sigmoid', weights=pretrained_weights))
    else:
        model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# >Train the model and measure performance
def train_and_evaluate(model, x_train, y_train, x_test, y_test, epochs=10):
    history = {'loss': [], 'accuracy': []}
    for epoch in range(epochs):
        start_time = time.time()
        history_epoch = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=64, verbose=0)
        end_time = time.time()
        history['loss'].append(history_epoch.history['loss'][0])
        history['accuracy'].append(history_epoch.history['val_accuracy'][0])
        print(f"Epoch {epoch + 1}/{epochs} - Time: {end_time - start_time:.2f}s - Val Accuracy: {history_epoch.history['val_accuracy'][0]:.4f}")
    return history

# %Load the pre-trained RBM model and extract weights
rbm_model = tf.keras.models.load_model('SupervisedwpreRBM.h5', custom_objects={'RBM': RBM})
pretrained_weights = [rbm_model.layers[0].weights[0].numpy(), rbm_model.layers[0].weights[1].numpy()]

#% Train the model with random initialization
print("\nTraining model with random initialization:")
random_model = create_supervised_model()
history_random = train_and_evaluate(random_model, x_train, y_train_one_hot, x_test, y_test_one_hot)

#% Train the model with pre-trained RBM weights
print("\nTraining model with pre-trained RBM weights:")
pretrained_model = create_supervised_model(pretrained_weights=pretrained_weights)
history_pretrained = train_and_evaluate(pretrained_model, x_train, y_train_one_hot, x_test, y_test_one_hot)

# %Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_random['accuracy'], label='Random Initialization', marker='o')
plt.plot(history_pretrained['accuracy'], label='Pre-trained RBM', marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_random['loss'], label='Random Initialization', marker='o')
plt.plot(history_pretrained['loss'], label='Pre-trained RBM', marker='o')
plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
