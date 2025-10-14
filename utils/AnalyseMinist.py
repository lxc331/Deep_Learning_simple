
class AnalyseMinist:
    def __init__(self, data_path='../DataSet/MNIST'):
        self.data_path = data_path
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None


    def analyse(self):
        # 训练集
        self.train_images, train_rows, train_cols = self._load_images(
            f'{self.data_path}/train-images-idx3-ubyte'
        )
        self.train_labels = self._load_labels(
            f'{self.data_path}/train-labels-idx1-ubyte'
        )

        # 测试集
        self.test_images, test_rows, test_cols = self._load_images(
            f'{self.data_path}/t10k-images-idx3-ubyte'
        )
        self.test_labels = self._load_labels(
            f'{self.data_path}/t10k-labels-idx1-ubyte'
        )
        print(self.train_images.shape, self.train_labels.shape)
        print(self.test_images.shape, self.test_labels.shape)
