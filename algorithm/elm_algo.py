import numpy as np

class ELM:
    def __init__(self, input_size, hidden_size, activation='sigmoid', C=0.1):
        """
        极限学习机模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层神经元数量
            activation: 激活函数类型，可选 'sigmoid', 'tanh', 'relu', 'sin'
            C: 正则化参数，值越大正则化效果越强
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.C = C  # 正则化参数
        self.weights = None
        self.bias = None
        self.beta = None

    def _activate(self, x):
        """根据选择的激活函数类型进行激活"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sin':
            return np.sin(x)
        else:
            return 1 / (1 + np.exp(-x))  # 默认使用sigmoid

    def fit(self, X, y):
        """
        训练ELM模型
        
        参数:
            X: 输入特征，形状为 (n_samples, input_size)
            y: 目标值，形状为 (n_samples,)
            
        返回:
            self: 训练好的模型
        """
        n_samples = X.shape[0]
        
        # 初始化输入层到隐藏层的参数 - 使用更好的初始化方法
        # 使用He初始化，适合ReLU等激活函数
        scale = np.sqrt(2.0 / self.input_size)
        self.weights = np.random.randn(self.input_size, self.hidden_size) * scale
        self.bias = np.random.randn(self.hidden_size) * 0.1
        
        # 计算隐藏层输出
        H = self._activate(np.dot(X, self.weights) + self.bias)
        
        # 添加正则化项进行训练 (Ridge Regression)
        # 计算输出权重 (β = (H^T·H + C·I)^(-1)·H^T·y)
        I = np.eye(self.hidden_size)
        if self.C > 0:
            # 带正则化的伪逆计算
            H_pinv = np.linalg.inv(H.T.dot(H) + self.C * I).dot(H.T)
        else:
            # 标准伪逆计算
            H_pinv = np.linalg.pinv(H)
            
        self.beta = np.dot(H_pinv, y)
        
        return self

    def predict(self, X):
        """预测新样本"""
        H = self._activate(np.dot(X, self.weights) + self.bias)
        return np.dot(H, self.beta)

    def score(self, X, y):
        """计算RMSE评分"""
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y_pred - y)**2))
    
    def r2_score(self, X, y):
        """计算R²评分"""
        y_pred = self.predict(X)
        ss_total = np.sum((y - y.mean()) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def generate_example_data(self, n_samples=200, noise_level=0.1, function='sin'):
        """
        生成示例数据
        
        参数:
            n_samples: 样本数量
            noise_level: 噪声水平
            function: 生成函数类型，可选 'sin', 'complex'
            
        返回:
            X: 输入特征
            y: 目标值
        """
        X = np.linspace(-10, 10, n_samples)[:, None]
        
        if function == 'sin':
            # 简单的正弦函数
            y = np.sin(X).ravel() + np.random.normal(0, noise_level, n_samples)
        elif function == 'complex':
            # 更复杂的函数：sin(x) + 0.1*x^2 + 0.01*x^3
            y = (np.sin(X) + 0.1 * X**2 + 0.01 * X**3).ravel() + np.random.normal(0, noise_level, n_samples)
        else:
            # 默认使用正弦函数
            y = np.sin(X).ravel() + np.random.normal(0, noise_level, n_samples)
            
        return X, y