import tensorflow as tf
from tensorflow.keras.layers import Layer


class LocalResponseNormalization(Layer):
    """
    局部响应归一化(LRN)层的TensorFlow实现

    参数:
        depth_radius: 半窗口大小，用于计算归一化(默认5)
        bias: 偏移项(默认1.0)
        alpha: 缩放因子(默认1e-4)
        beta: 指数项(默认0.75)
    """

    def __init__(self, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75, **kwargs):
        super(LocalResponseNormalization, self).__init__(**kwargs)
        self.depth_radius = depth_radius
        self.bias = bias
        self.alpha = alpha
        self.beta = beta

        # 保存配置用于序列化
        self.config = {
            'depth_radius': depth_radius,
            'bias': bias,
            'alpha': alpha,
            'beta': beta
        }

    def call(self, inputs):
        """
        前向传播计算

        参数:
            inputs: 输入张量，形状为[batch, height, width, channels]

        返回:
            归一化后的张量
        """
        return tf.nn.local_response_normalization(
            inputs,
            depth_radius=self.depth_radius,
            bias=self.bias,
            alpha=self.alpha,
            beta=self.beta
        )

    def get_config(self):
        """获取层配置，用于模型保存和加载"""
        base_config = super(LocalResponseNormalization, self).get_config()
        return dict(list(base_config.items()) + list(self.config.items()))

    def compute_output_shape(self, input_shape):
        """计算输出形状（与输入相同）"""
        return input_shape


# 使用示例
if __name__ == "__main__":
    # 创建示例数据（模拟CNN特征图）
    batch_size, height, width, channels = 4, 32, 32, 64
    example_input = tf.random.normal([batch_size, height, width, channels])

    # 创建LRN层
    lrn_layer = LocalResponseNormalization(
        depth_radius=5,
        bias=1.0,
        alpha=1e-4,
        beta=0.75
    )

    # 应用LRN
    output = lrn_layer(example_input)

    print(f"输入形状: {example_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"LRN参数: depth_radius={lrn_layer.depth_radius}, "
          f"bias={lrn_layer.bias}, alpha={lrn_layer.alpha}, beta={lrn_layer.beta}")