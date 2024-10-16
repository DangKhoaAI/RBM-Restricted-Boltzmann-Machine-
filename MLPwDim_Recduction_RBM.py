import tensorflow as tf
import numpy as np
from preprocess_RBM import *
#? class mô hình MLP , fuction load và giảm chiều dữ liệu từ RBM , function load mnist data
#>class Mô hình MLP
class MLP:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.model = tf.keras.Sequential()

        # Thêm các lớp ẩn
        for hidden_dim in hidden_dims:
            self.model.add(tf.keras.layers.Dense(hidden_dim, activation='relu'))

        # Lớp đầu ra
        self.model.add(tf.keras.layers.Dense(output_dim, activation='softmax'))

        # Compile mô hình với loss và optimizer
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, train_data, train_labels, batch_size=64, epochs=10):
        self.model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)

    def evaluate(self, test_data, test_labels):
        loss, accuracy = self.model.evaluate(test_data, test_labels)
        print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

#% thực hiện train
if __name__ == "__main__":
    #% Load dữ liệu
    x_train, y_train, x_test, y_test = load_mnist_data()

    #% Giảm chiều dữ liệu bằng RBM
    reduced_train_data = preprocess_data_with_rbm(x_train)
    reduced_test_data = preprocess_data_with_rbm(x_test)

    #% Xây dựng và huấn luyện MLP với dữ liệu đã được giảm chiều
    mlp = MLP(input_dim=128, hidden_dims=[128, 64], output_dim=10)
    mlp.train(reduced_train_data, y_train, batch_size=64, epochs=10)

    #% Đánh giá trên tập dữ liệu test
    mlp.evaluate(reduced_test_data, y_test)
