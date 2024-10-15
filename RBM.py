import numpy as np
import os
import tensorflow as tf
#? file này là class thuật toán RBM cùng hàm trainin
class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        # Khởi tạo trọng số và bias
        self.W = tf.Variable(tf.random.normal([self.n_visible, self.n_hidden], stddev=0.01), name="weights")
        self.h_bias = tf.Variable(tf.zeros([self.n_hidden]), name="hidden_bias")
        self.v_bias = tf.Variable(tf.zeros([self.n_visible]), name="visible_bias")

    def sample_prob(self, probs):
        """Lấy mẫu nhị phân dựa trên xác suất"""
        return tf.cast(tf.random.uniform(tf.shape(probs)) < probs, dtype=tf.float32)

    def forward(self, visible):
        """Tính xác suất hidden"""
        hidden_probs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.h_bias)
        hidden_sample = self.sample_prob(hidden_probs)
        return hidden_probs, hidden_sample

    def backward(self, hidden):
        """Tính xác suất visible"""
        visible_probs = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.W)) + self.v_bias)
        visible_sample = self.sample_prob(visible_probs)
        return visible_probs, visible_sample

    def contrastive_divergence(self, v0):
        """Thuật toán học Contrastive Divergence"""
        # Phase 1: forward pass (positive phase)
        h0_prob, h0_sample = self.forward(v0)
        
        # Phase 2: backward pass (negative phase)
        v1_prob, v1_sample = self.backward(h0_sample)
        h1_prob, h1_sample = self.forward(v1_sample)

        # Cập nhật trọng số và bias
        positive_grad = tf.matmul(tf.transpose(v0), h0_prob)
        negative_grad = tf.matmul(tf.transpose(v1_sample), h1_prob)

        dW = positive_grad - negative_grad
        dv_bias = tf.reduce_mean(v0 - v1_sample, axis=0)
        dh_bias = tf.reduce_mean(h0_prob - h1_prob, axis=0)

        self.W.assign_add(self.learning_rate * dW)
        self.v_bias.assign_add(self.learning_rate * dv_bias)
        self.h_bias.assign_add(self.learning_rate * dh_bias)

    def reconstruct(self, visible):
        """Tái tạo lại dữ liệu từ visible"""
        hidden_probs, hidden_sample = self.forward(visible)
        visible_probs, visible_sample = self.backward(hidden_sample)
        return visible_probs
    def save_model(self, checkpoint_dir='rbm_checkpoint'):
        checkpoint = tf.train.Checkpoint(weights=self.W, h_bias=self.h_bias, v_bias=self.v_bias)
        checkpoint.save(file_prefix=os.path.join(checkpoint_dir, "rbm"))

    # Tải mô hình
    def load_model(self, checkpoint_dir='rbm_checkpoint'):
        checkpoint = tf.train.Checkpoint(weights=self.W, h_bias=self.h_bias, v_bias=self.v_bias)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"Model restored from {latest_checkpoint}")
        else:
            print("No checkpoint found.")

#>Hàm training RBM và saved model 
def train_rbm(rbm, data, batch_size=64, epochs=10,checkpoint_dir='rbm_checkpoint'):
    num_samples = data.shape[0]
    for epoch in range(epochs):
        np.random.shuffle(data)
        for i in range(0, num_samples, batch_size):
            batch = data[i:i+batch_size]
            rbm.contrastive_divergence(batch)
        print(f"Epoch {epoch+1} completed")
    rbm.save_model(checkpoint_dir)
    print("Model saved.")

