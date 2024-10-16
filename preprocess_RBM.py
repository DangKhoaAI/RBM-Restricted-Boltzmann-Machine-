import tensorflow as tf
from RBM import RBM
from RBM_sklearn.RBM_scikitlearn import RBMsklearn
#? preprocessing data (dim reduction) by RBM 
#> function load và giảm chiều dữ liệu bằng RBM
def preprocess_data_with_rbm(data, rbm_model_path='rbm_checkpoint128'):
    #_> Khởi tạo RBM với thông số đã lưu
    data = tf.cast(data, dtype=tf.float32)
    n_visible = data.shape[1]  # Số lượng đặc trưng của dữ liệu gốc (784)
    n_hidden = 128  # Sử dụng n_hidden đã lưu trong model
    rbm = RBM(n_visible=n_visible, n_hidden=n_hidden)
    #_> Tải mô hình RBM đã lưu
    rbm.load_model(checkpoint_dir=rbm_model_path)    
    #_> Dùng RBM để giảm chiều dữ liệu
    _, hidden_representation = rbm.forward(data)
    return hidden_representation.numpy()
#> function load và giảm chiều dữ liệu bằng RBMsklearn
def preprocess_data_with_rbmsklearn(data, rbm_model_path='RBM_sklearn\\rbm_model.pkl'):
    #_> Khởi tạo RBM với thông số đã lưu
    data = tf.cast(data, dtype=tf.float32)
    n_visible = 784  # Number of pixels in MNIST image
    n_hidden = 256   # Number of hidden neurons
    rbm = RBMsklearn(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.01)
    #% Load model from checkpoint
    rbm.load_model(rbm_model_path)   
    #_> Dùng RBM để giảm chiều dữ liệu
    hidden_activations = rbm.transform(data)
    return hidden_activations
#> Function Load dữ liệu và tiền xử lý MNIST
def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28) / 255.0  # 784 feature
    x_test = x_test.reshape(-1, 28 * 28) / 255.0
    return x_train, y_train, x_test, y_test