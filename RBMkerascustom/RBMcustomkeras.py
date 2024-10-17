import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
#? class mô hình RBM và function training RBM (custom keras )
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