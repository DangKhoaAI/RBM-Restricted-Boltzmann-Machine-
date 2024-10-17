import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Sequential
from DBN_deepbeliefnetwork import *

# > Tạo mô hình MLP
def create_mlp_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Chuyển đổi hình ảnh 28x28 thành vector 784
        Dense(256, activation='sigmoid'),   # Lớp ẩn đầu tiên
        Dense(128, activation='sigmoid'),    # Lớp ẩn thứ hai
        Dense(10, activation='softmax')   # Lớp đầu ra với 10 neuron (0-9)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# > Huấn luyện MLP trên dữ liệu MNIST
def train_mlp(x_train, y_train, x_test, y_test):
    model = create_mlp_model()
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
    return model

# > Lấy trọng số từ MLP và áp dụng cho DBN
def transfer_weights_to_rbm(rbm, mlp_layer):
    weights, biases = mlp_layer.get_weights()
    print(weights.shape ,biases.shape)
    # Chuyển đổi trọng số từ lớp Dense sang RBM
    rbm.W.assign(weights)  # Trọng số từ lớp MLP sang RBM
    rbm.h_bias.assign(biases)  # Áp dụng bias cho lớp ẩn của RBM
    rbm.v_bias.assign(np.zeros(rbm.W.shape[0]))  # Thiết lập bias đầu ra bằng 0

# > Chuyển trọng số vào DBN
def transfer_weights_to_dbs(rbm_model, mlp_model):
    mlp_layers = [layer for layer in mlp_model.layers if isinstance(layer, Dense)][:2]
    rbm_layers = [layer for layer in rbm_model.layers if isinstance(layer, RBM_dbn)]
    
    for rbm, mlp_layer in zip(rbm_layers, mlp_layers):
        transfer_weights_to_rbm(rbm, mlp_layer)   # Chuyển weights từ MLP sang RBM

if __name__ == "__main__":
    # * Huấn luyện MLP
    # > Dữ liệu training
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train_flat = x_train.reshape(x_train.shape[0], -1) #>cho finetuning DBN sau này
    #% Huấn luyện MLP
    model_file = 'model_mlp.h5'
    
    # Kiểm tra xem model_mlp.h5 đã tồn tại chưa
    if os.path.exists(model_file):
        print(f"Loading existing MLP model from {model_file}.")
        mlp_model = tf.keras.models.load_model(model_file)
    else:
        # Nếu không tồn tại, huấn luyện mô hình
        print("Training new MLP model.")
        mlp_model = train_mlp(x_train, y_train, x_test, y_test)
        mlp_model.save(model_file)
        print(f"MLP model saved at {model_file}")

    # * Transfer sang DBN
    #% Tải mô hình MLP đã huấn luyện
    mlp_model = tf.keras.models.load_model('model_mlp.h5')
    #% tạo kiến trúc DBN
    # Tạo kiến trúc model
    n_hidden1 = 256  # Số lượng neuron ẩn cho lớp RBM đầu tiên
    n_hidden2 = 128  # Số lượng neuron ẩn cho lớp RBM thứ hai
    rbm1 = RBM_dbn(n_hidden=n_hidden1, learning_rate=0.01, name="RBM_1")
    rbm2 = RBM_dbn(n_hidden=n_hidden2, learning_rate=0.01, name="RBM_2")

    inputs = Input(shape=(784,))
    x = rbm1(inputs)
    outputs = rbm2(x)
    model = tf.keras.Model(inputs, outputs)
    model.build((None, 784))
    
    # > Chuyển trọng số vào DBN
    transfer_weights_to_dbs(model, mlp_model)
    print("Weights transferred from MLP to DBN.")
    train_rbm(model, x_train_flat, epochs=5)
    model.save('reverted_dbn.h5')
