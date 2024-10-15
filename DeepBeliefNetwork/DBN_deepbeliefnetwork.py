import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
#? triển khai class RBM , tạo function train DBN , tạo model train DBN
#> Định nghĩa lớp RBM
@tf.keras.utils.register_keras_serializable()
class RBM(tf.keras.Model):
    def __init__(self, n_hidden, learning_rate=0.01,  **kwargs):
        super(RBM, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.W = None          
        self.h_bias = None
        self.v_bias = None

    def build(self, input_shape):
        n_visible = input_shape[-1]  

        self.W = self.add_weight(shape=(n_visible, self.n_hidden),
                                 initializer='random_normal',
                                 trainable=True,
                                 name="weights")
        self.h_bias = self.add_weight(shape=(self.n_hidden,),
                                       initializer='zeros',
                                       trainable=True,
                                       name="hidden_bias")
        self.v_bias = self.add_weight(shape=(n_visible,),
                                       initializer='zeros',
                                       trainable=True,
                                       name="visible_bias")
        
        super().build(input_shape)  # Gọi super().build(input_shape) để hoàn thành việc xây dựng
        
    def get_config(self):
        return {
            "n_hidden": self.n_hidden,
            "learning_rate": self.learning_rate
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    #>HÀM CALL SỬA (BÊN RBM LÀ reconstruct)
    def call(self, inputs):
        # Thay vì khôi phục đầu vào, chỉ cần trả về hidden_sample hoặc hidden_probs
        hidden_probs, hidden_sample = self.forward(inputs)
        return hidden_sample

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    def forward(self, visible):
        hidden_probs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.h_bias)
        hidden_sample = self.sample_prob(hidden_probs)
        return hidden_probs, hidden_sample

    def backward(self, hidden):
        visible_probs = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.W)) + self.v_bias)
        visible_sample = self.sample_prob(visible_probs)
        return visible_probs, visible_sample

    @tf.function
    def contrastive_divergence(self, v0):
        h0_prob, h0_sample = self.forward(v0)
        v1_prob, v1_sample = self.backward(h0_sample)
        h1_prob, h1_sample = self.forward(v1_sample)

        positive_grad = tf.matmul(tf.transpose(v0), h0_prob)
        negative_grad = tf.matmul(tf.transpose(v1_sample), h1_prob)

        dW = positive_grad - negative_grad
        dv_bias = tf.reduce_mean(v0 - v1_sample, 0)
        dh_bias = tf.reduce_mean(h0_prob - h1_prob, 0)

        self.W.assign_add(self.learning_rate * dW)
        self.v_bias.assign_add(self.learning_rate * dv_bias)
        self.h_bias.assign_add(self.learning_rate * dh_bias)
        return self.W, self.v_bias, self.h_bias

    def reconstruct(self, visible):
        hidden_probs, hidden_sample = self.forward(visible)
        visible_probs, visible_sample = self.backward(hidden_sample)
        return visible_probs

#> Định nghĩa hàm train model (train class RBM của model)
def train_rbm(model, data, batch_size=64, epochs=10):
    num_samples = data.shape[0]
    for epoch in range(epochs):
        np.random.shuffle(data)
        for i in range(0, num_samples, batch_size):
            batch = data[i:i + batch_size]
            # Huấn luyện từng lớp RBM
            for rbm in model.layers:  
                if isinstance(rbm, RBM):  # Kiểm tra xem lớp hiện tại có phải là RBM không
                    rbm.contrastive_divergence(batch)
                    batch = rbm(batch)  # Chuyển đổi đầu ra của RBM cho lớp tiếp theo
        print(f"Epoch {epoch + 1} completed")

if __name__ == "__main__":
    # >Dữ liệu training
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)

    #% Tạo kiến trúc model
    n_hidden1 = 64  # Số lượng neuron ẩn cho lớp RBM đầu tiên
    n_hidden2 = 32  # Số lượng neuron ẩn cho lớp RBM thứ hai
    rbm1 = RBM(n_hidden=n_hidden1, learning_rate=0.01, name="RBM_1")
    rbm2 = RBM(n_hidden=n_hidden2, learning_rate=0.01, name="RBM_2")

    inputs = Input(shape=(784,))
    x = rbm1(inputs)
    outputs = rbm2(x)
    model = tf.keras.Model(inputs, outputs)
    model.build((None, 784))
    #% Huấn luyện model
    train_rbm(model, x_train, epochs=12)
    model.save('model_dbn.h5')
    print("Save model at model_dbn.h5")
