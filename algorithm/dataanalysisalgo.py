# 数据处理
import pandas as pd
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer


def readfile(path):
    df = pd.read_csv(path, index_col=0).reset_index(drop=True)
    return df


def calcpca(df):
    # 创建PCA模型，并指定要保留的主成分数量
    pca = PCA(n_components=1)
    # 拟合模型并转换数据
    X_pca = pca.fit_transform(df)
    # 查看主成分的方差贡献比
    #print(pca.explained_variance_ratio_)
    return X_pca


def calfactor(df):
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    # KMO检验
    from factor_analyzer.factor_analyzer import calculate_kmo
    kmo_all, kmo_model = calculate_kmo(df)

    faa = FactorAnalyzer(25, rotation=None)
    faa.fit(df)

    # 得到特征值ev、特征向量v
    ev, v = faa.get_eigenvalues()
    # print(ev,v)
    return ev, v

import numpy as np
import pandas as pd
from scipy.spatial.distance import chebyshev, jensenshannon


def readfile(data_path):
    df = pd.read_csv(data_path)
    return df

def fusion(a, b):
    m1 = np.array(a)
    m2 = np.array(b)
    k = np.sum(np.outer(m1, m2))  # 计算冲突因子 k，避免重复计算
    res = np.sum(m1 * m2)  # 计算相同元素的乘积之和
    k = k - res
    A = m1 * m2 / (1 - k)  # 计算似真度函数
    P = A / np.sum(A)  # 计算融合结果
    return P


def dsth(data):
    datacopy = data.copy()
    for i in range(len(data) - 1):
        datacopy[i + 1] = fusion(datacopy[i], datacopy[i + 1])
    result = datacopy[-1]
    return result


def yager_ds(data):  # 各证据来源的权重（alpha值）模糊度
    k = 0
    result = np.zeros(data.shape[1] + 1)
    for i in range(data.shape[1]):
        product = np.prod(data[:, i])  # i列乘积
        k += product
        result[i] = product
    k = 1 - k
    result[-1] = k
    return result


def sugeno_ds(data):
    k = 0
    numbers = np.sum(np.arange(1, data.shape[0]))  # 求kij
    k_ = np.zeros(numbers)
    result = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        product = np.prod(data[:, i])  # i列乘积
        k += product
        for j in range(i + 1, data.shape[1]):
            k_[i + j - 1] = 1 - np.inner(data[j, :], data[i, :])
    k = 1 - k
    k1 = np.sum(k_) / 3
    ɛ = 1 / (np.exp(k1))
    for i in range(data.shape[1]):
        product = np.prod(data[:, i])  # i列乘积
        result[i] = product + ɛ * k * np.sum(data[:, i]) / 3
    return result


#  切比雪夫距离改进的证据理论
def chebyshev_ds(data):
    lendata = data.shape[1]
    d = np.zeros((lendata, lendata))  # 证据体距离矩阵
    for i in range(data.shape[1] - 1):
        for j in range(data.shape[1] - 1):
            distance = chebyshev(data[:, i], data[:, j + 1])
            d[i, j + 1] = distance
            d[j + 1, i] = distance
    # 证据体间的相似度
    s = 1 - d
    # 计算可信度
    crd = np.zeros(lendata)
    for i in range(lendata):
        crd[i] = (np.sum(s[i]) - 1) / (np.sum(s) - lendata)  # i是对角线和
    # 可信度分配给证据体
    mw = data * crd[np.newaxis, :]
    result = dsth(mw)
    return result


def murphy_ds(data):
    data_len = data.shape[0]
    result = np.zeros(data_len)
    avg_bpa = np.zeros(data_len)
    for i in range(data_len):
        avg_bpa[i] = np.mean(data[:, i])
    for i in range(data_len):
        result[i] = avg_bpa[i] ** 2 / np.sum(avg_bpa ** 2)  # 迭代两次
    return result


def dengyong_ds(data):
    lendata = data.shape[1]
    d = np.zeros((lendata, lendata))  # 巴拿赫-普列哈特距离#### Jensen-Shannon公式？
    for i in range(data.shape[1] - 1):
        for j in range(data.shape[1] - 1):
            distance = jensenshannon(data[:, i], data[:, j + 1])
            d[i, j + 1] = distance
            d[j + 1, i] = distance
    # 证据体间的相似度
    s = 1 - d
    # 计算可信度
    crd = np.zeros(lendata)
    for i in range(lendata):
        crd[i] = (np.sum(s[i]) - 1) / (np.sum(s) - lendata)  # i是对角线和
    # 可信度分配给证据体
    mw = data * crd[np.newaxis, :]
    result = dsth(mw)
    return result


def gaijin_ds(data):
    p = 4
    dataminus = 0
    data_len = data.shape[0]
    result = np.zeros(data_len + 1)
    d = np.zeros((data_len, data_len))  # data.shape[0]是行数
    for i in range(data_len):
        for j in range(i + 1, data_len):
            dataminus = data[i, :] - data[j, :]
            d[i, j] = (np.sum(dataminus ** p)) ** (1 / 4)
    con = np.exp(-d)
    con[con == 1] = 0
    crd = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        crd[i] = (np.sum(con[i, :]) + np.sum(con[:, i]) - con[i, i]) / (2 * np.sum(con))
    for i in range(data.shape[1]):
        result[i] = np.sum(crd * data[:, i])
    k = 1-np.sum(result**3)
    result[-1] = k
    return result


def calds(df):
    df = df.to_numpy()
    dsth_result = dsth(df)
    chebyshev_result = chebyshev_ds(df)
    yager_result = yager_ds(df)
    sugeno_result = sugeno_ds(df)
    murphy_result = murphy_ds(df)
    dengyong_result = dengyong_ds(df)
    gaijin_result = gaijin_ds(df)

    index = ['传统D-S', '切比雪夫D-S', 'yagerD-S', 'sugenoD-S', 'murphyD-S', 'dengyongD-S', '改进D-S']
    pandasresult = pd.DataFrame(
        [dsth_result, chebyshev_result, yager_result, sugeno_result, murphy_result,
         dengyong_result, gaijin_result])
    pandasresult = pandasresult.round(3)
    pandasresult.insert(loc=0, column='名称', value=index)
    return pandasresult

