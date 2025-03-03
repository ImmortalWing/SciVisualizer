import cv2
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def crackidentification(imagepath):
    image_color = cv2.imread(imagepath)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

    # 自动搜索最优h值
    h_values = list(range(0, 41, 5))  # h搜索范围0-40，步长1
    variances = []
    
    # 遍历h值计算方差
    for h in h_values:
        denoised = cv2.fastNlMeansDenoising(image_gray, h=h, templateWindowSize=7, searchWindowSize=21)
        laplacian = cv2.Laplacian(denoised, cv2.CV_64F)
        variances.append(np.var(laplacian))
    
    # GPR拟合曲线
    x_data = np.array(h_values).reshape(-1, 1)
    y_data = np.array(variances).reshape(-1, 1)
    
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr_model.fit(x_data, y_data)
    
    # 计算导数
    x_fit = np.linspace(min(x_data), max(x_data), 100).reshape(-1, 1)
    y_pred_mean, _ = gpr_model.predict(x_fit, return_std=True)
    dy = np.diff(y_pred_mean.flatten())
    derivative = dy / (x_fit[1] - x_fit[0])[0]
    
    # 寻找导数第二次大于-10的h值
    threshold = -20
    cross_points = []
    for i in range(1, len(derivative)):
        if derivative[i-1] <= threshold and derivative[i] > threshold:
            cross_points.append(x_fit[i][0])
    
    optimal_h = int(round(cross_points[1])) if len(cross_points) > 1 else 35
    # 使用最优h值进行去噪
    dst = cv2.fastNlMeansDenoising(image_gray, h=optimal_h, templateWindowSize=7, searchWindowSize=21)

    image_homomorphic_filter = homomorphic_filter(dst,d0=10, rl=0.9, rh=2.5, c=1, h=2, l=0.5)
    dilated_image = dilating(image_homomorphic_filter)
    areathreshold_img = filtering(dilated_image)
    thinned_image = cv2.ximgproc.thinning(areathreshold_img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    plt.imshow(thinned_image, cmap='gray')
    plt.axis('off')
    height, width = image_color.shape[:2]
    plt.savefig('data/crackidentifyresult.png', 
               bbox_inches='tight', 
               pad_inches=0,
               dpi=width/5)
    return('data/crackidentifyresult.png')

def homomorphic_filter(src, d0=10, rl=1, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    gray = np.log(1e-5 + gray)  # 取对数
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)  # FFT傅里叶变换
    gray_fftshift = np.fft.fftshift(gray_fft)  # FFT中心化
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M ** 2 + N ** 2)  # 计算距离
    Z = (rh - rl) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + rl  # H(u,v)传输函数
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)  # IFFT逆傅里叶变换
    dst = np.real(dst_ifft)  # IFFT取实部
    dst = np.exp(dst) - 1  # 还原
    dst = np.uint8(np.clip(dst, 0, 255))
    return dst


def dilating(image_homomorphic_filter):
    _, binary_image = cv2.threshold(image_homomorphic_filter * 255, 127, 255, cv2.THRESH_BINARY)
    # 定义结构元素，这里使用一个3x3的矩形结构元素
    # 创建一个5x5的椭圆形结构元素
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # 执行膨胀操作
    dilated_image = cv2.dilate(binary_image, ellipse_kernel, iterations=5)
    return dilated_image

def filtering(dilated_image):
    # 检测轮廓
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算自适应面积阈值（例如取轮廓面积的平均值）
    areas = [cv2.contourArea(cnt) for cnt in contours]
    mean_area = np.mean(areas)

    # 设定阈值比例（可以根据需求调整）
    area_threshold = 1 * mean_area

    # 创建一个空白图像用于绘制符合面积阈值的轮廓
    output = np.zeros(dilated_image.shape[:2], dtype='uint8')  # 仅取宽高两维

    # 计算轮廓的外接矩形长宽比
    def aspect_ratio(contour):
        x, y, w, h = cv2.boundingRect(contour)
        return float(w) / h if h > 0 else 0


    # 筛选并绘制符合条件的轮廓
    for cnt in contours:
        area = cv2.contourArea(cnt)
        ratio = aspect_ratio(cnt)
        if area >= area_threshold :#and (ratio>1 or ratio<0.5):  # 保留面积大于阈值的轮廓
            cv2.drawContours(output, [cnt], -1, (255), thickness=cv2.FILLED)

    areathreshold_img = output
    return areathreshold_img
