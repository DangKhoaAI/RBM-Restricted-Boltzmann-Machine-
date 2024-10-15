import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam
from RBMcustomkeras import RBM  # Import the RBM class
#? file này tạo thuật toán supervised VỚI các weight đã pretraing từ RBM
#> Load the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train_binary = (x_train > 0.5).astype(np.float32)
x_test_binary = (x_test > 0.5).astype(np.float32)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

#> Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#>Load the pre-trained RBM model
rbm_model = tf.keras.models.load_model('RBMmodel.h5', custom_objects={'RBM': RBM})

#> Define the supervised model with pre-trained RBM weights in Sequential format
supervised_model = Sequential([
    InputLayer(input_shape=(784,)),
    Dense(128, activation='sigmoid'),# weights=[rbm_model.layers[-1].W.numpy(), rbm_model.layers[-1].h_bias.numpy()]),  # Pre-trained RBM weights
    Dense(64, activation='relu'),  # Additional hidden layer
    Dense(10, activation='softmax')  # Output layer for 10 classes (MNIST)
])

#> Compile the model
supervised_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#> Train the supervised model
supervised_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

#> Evaluate the model
loss, accuracy = supervised_model.evaluate(x_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

#> Save the trained supervised model
supervised_model.save('SupervisedwpreRBM.h5')
