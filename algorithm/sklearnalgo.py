import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def svm():
    # 加载鸢尾花数据集作为示例数据
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 仅使用前两个特征以便可视化
    y = iris.target

    # 将数据集分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建SVM分类器
    svm_classifier = SVC(kernel='linear', C=1.0)  # 线性核函数

    # 训练模型
    svm_classifier.fit(X_train, y_train)

    # 预测
    y_pred = svm_classifier.predict(X_test)

    # 计算准确率
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"准确率: {accuracy:.2f}%")

    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svm_classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('支持向量机分类结果')
    plt.show()

def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # 加载示例数据集（这里以鸢尾花数据集为例）
    data = load_iris()
    X = data.data
    y = data.target

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建随机森林分类器
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练模型
    rf_classifier.fit(X_train, y_train)

    # 预测测试集
    y_pred = rf_classifier.predict(X_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    print(f"随机森林模型准确率: {accuracy:.2f}")


def elm():
    from sklearn.neural_network import MLPRegressor
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    housing = fetch_california_housing()
    # 加载示例数据集（这里以波士顿房价数据集为例）
    data = housing
    print(data)
    X = data.data
    y = data.target

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建极限学习机回归器
    elm_regressor = MLPRegressor(hidden_layer_sizes=(100,), activation='logistic', max_iter=1000, random_state=42)

    # 训练模型
    elm_regressor.fit(X_train, y_train)

    # 预测测试集
    y_pred = elm_regressor.predict(X_test)

    # 评估模型性能
    mse = mean_squared_error(y_test, y_pred)
    print(f"极限学习机模型均方误差: {mse:.2f}")

def rbf():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)  # 添加一些噪声

    # 定义径向基函数
    def rbf(x, c, s):
        return np.exp(-1 / (2 * s ** 2) * (x - c) ** 2)

    # 定义径向基函数网络
    def rbf_network(x, centers, width):
        return np.array([rbf(x, c, width) for c in centers])

    # 选择一些中心点和宽度
    centers = np.linspace(0, 10, 10)
    width = 0.5

    # 计算径向基函数网络的输出
    phi = rbf_network(x, centers, width)

    # 使用线性回归拟合
    w = np.linalg.lstsq(phi.T, y, rcond=None)[0]

    # 预测
    x_pred = np.linspace(0, 10, 1000)
    phi_pred = rbf_network(x_pred, centers, width)
    y_pred = np.dot(phi_pred.T, w)

    # 绘制结果
    plt.scatter(x, y, label='实际数据')
    plt.plot(x_pred, y_pred, 'r', label='拟合结果')
    plt.legend()
    plt.show()



rbf()