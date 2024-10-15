import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt

#> Định nghĩa lớp RBM
@tf.keras.utils.register_keras_serializable()
class RBM(tf.keras.Model):
    def __init__(self, n_hidden, learning_rate=0.01, **kwargs):
        super(RBM, self).__init__(**kwargs, name='customRBM')
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.W = None          
        self.h_bias = None
        self.v_bias = None

    def build(self, input_shape):
        n_visible = input_shape[-1]  # Lấy số lượng đơn vị hiển thị từ hình dạng đầu vào

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
    
    def call(self, inputs):
        output = self.reconstruct(inputs)
        return output

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
            batch = data[i:i+batch_size]
            model.layers[-1].contrastive_divergence(batch)  # Truy cập lớp RBM trong model Keras
        print(f"Epoch {epoch+1} completed")
#> định nghĩa hàm visualize model
def visualize_reconstruction(model, data, index=0):
    original_img = data[index].reshape(1, -1)
    reconstructed_img = model.predict(original_img)

    original_img = original_img.reshape(28, 28)
    reconstructed_img = reconstructed_img.reshape(28, 28)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Ảnh gốc
    axes[0].imshow(original_img, cmap="gray")
    axes[0].set_title("Ảnh gốc")
    axes[0].axis('off')

    # Ảnh tái tạo
    axes[1].imshow(reconstructed_img, cmap="gray")
    axes[1].set_title("Ảnh tái tạo")
    axes[1].axis('off')

    plt.show()
if __name__=="__main__":
    # Dữ liệu training
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)

    #%Tạo kiến trúc model
    n_hidden = 64  # Số lượng neuron ẩn
    rbm = RBM(n_hidden=n_hidden, learning_rate=0.01)
    inputs = Input(shape=(784,))
    outputs=rbm(inputs)
    model = tf.keras.Model(inputs,outputs)
    model.build((None,784))
    #%train model
    train_rbm(model, x_train, epochs=20)
    model.save('model_rbm.h5')
    print("Save model at model_rbm.h5")

    # Load mô hình từ file .h5
    model_reloaded = tf.keras.models.load_model('model_rbm.h5', custom_objects={'RBM': RBM})
    print("Model reloaded from  model_rbm.h5")

    visualize_reconstruction(model, x_train, index=0)
    visualize_reconstruction(model_reloaded, x_train, index=0)
    
    """
    #% lưu tham số model
    model.save_weights("rbm.weights.h5")
    #%load lại model
    model_reloaded = tf.keras.Model(inputs, rbm(inputs))
    model_reloaded.build((None, 784))
    model_reloaded.load_weights("rbm.weights.h5")

    visualize_reconstruction(model, x_train, index=0)
    visualize_reconstruction(model_reloaded, x_train, index=0)
    """