import tensorflow as tf
import numpy as np
#?: xem một vài loại RBM: BB-RBM , GB-RBM
class SimpleRBM:
    def __init__(self, visible_units, hidden_units, sigma=0.1):
        # Khởi tạo trọng số và bias
        self.W = tf.Variable(tf.random.normal([visible_units, hidden_units], stddev=0.01), dtype=tf.float32)
        self.h_bias = tf.Variable(tf.zeros([hidden_units]), dtype=tf.float32)
        self.sigma = sigma  # Độ lệch chuẩn cho phân phối Gauss

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    def forward_sample(self, visible):
        # Tính toán xác suất ẩn
        hidden_probs = tf.nn.sigmoid(tf.matmul(visible, self.W) + self.h_bias)
        # Lấy mẫu từ xác suất
        hidden_sample = self.sample_prob(hidden_probs)
        return hidden_sample

    def forward_gauss(self, visible):
        # Tính mean cho lớp ẩn
        mean_h = tf.matmul(visible, self.W) + self.h_bias
        hidden_sample = mean_h + self.sigma * tf.random.normal(tf.shape(mean_h))
        return mean_h, hidden_sample

# Hàm chính để thử nghiệm
if __name__ == "__main__":
    # Thiết lập tham số
    visible_units = 6
    hidden_units = 3

    # Tạo mô hình RBM
    rbm = SimpleRBM(visible_units, hidden_units)

    # Tạo một tensor đầu vào ngẫu nhiên
    visible_input = tf.random.uniform((5, visible_units), 0, 1)  # 5 mẫu với 6 đơn vị hiển thị

    # Thực hiện bước forward mẫu
    hidden_output_sample = rbm.forward_sample(visible_input)

    # Thực hiện bước forward Gauss
    mean_h, hidden_output_gauss = rbm.forward_gauss(visible_input)

    # In kết quả
    print("Visible Input:\n", visible_input.numpy())
    print("Hidden Output (Sample):\n", hidden_output_sample.numpy())
    print("Mean Hidden Output (Gauss):\n", mean_h.numpy())
    print("Hidden Output (Gauss Sample):\n", hidden_output_gauss.numpy())
