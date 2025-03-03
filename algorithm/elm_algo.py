import numpy as np

class ELM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights = None
        self.bias = None
        self.beta = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # 初始化输入层到隐藏层的参数
        self.weights = np.random.randn(self.input_size, self.hidden_size)
        self.bias = np.random.randn(self.hidden_size)
        
        # 计算隐藏层输出
        H = self._sigmoid(np.dot(X, self.weights) + self.bias)
        
        # 计算输出权重
        H_pinv = np.linalg.pinv(H)
        self.beta = np.dot(H_pinv, y)
        
        return self

    def predict(self, X):
        H = self._sigmoid(np.dot(X, self.weights) + self.bias)
        return np.dot(H, self.beta)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y_pred - y)**2))