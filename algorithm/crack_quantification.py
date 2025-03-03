import cv2
import numpy as np
from scipy import ndimage
import os

class CrackQuantification:
    """裂隙图片量化分析类"""
    
    def __init__(self):
        """初始化裂隙量化分析类"""
        self.image = None
        self.binary_image = None
        self.skeleton = None
        self.results = {}
    
    def load_image(self, image_path):
        """
        加载图片
        
        参数:
            image_path: 图片路径
        
        返回:
            成功加载返回True，否则返回False
        """
        if not os.path.exists(image_path):
            print(f"错误：图片路径 {image_path} 不存在")
            return False
            
        try:
            self.image = cv2.imread(image_path)
            if self.image is None:
                print(f"错误：无法读取图片 {image_path}")
                return False
                
            # 转换为灰度图
            if len(self.image.shape) == 3:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                
            return True
        except Exception as e:
            print(f"加载图片时出错: {str(e)}")
            return False
    
    def preprocess_image(self, threshold_value=127, blur_size=5):
        """
        预处理图片：去噪、二值化
        
        参数:
            threshold_value: 二值化阈值
            blur_size: 高斯模糊核大小
        """
        if self.image is None:
            print("错误：请先加载图片")
            return False
            
        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(self.image, (blur_size, blur_size), 0)
        
        # 检查图像平均亮度，决定是否需要反转
        avg_brightness = np.mean(blurred)
        
        # 二值化 - 确保裂隙是白色(255)，岩体是黑色(0)
        if avg_brightness > 127:
            # 如果图像整体偏亮，裂隙可能是暗色的，使用THRESH_BINARY_INV
            _, self.binary_image = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
            binary_method = "THRESH_BINARY_INV"
        else:
            # 如果图像整体偏暗，裂隙可能是亮色的，使用THRESH_BINARY
            _, self.binary_image = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
            binary_method = "THRESH_BINARY"
        
        # 检查二值化结果是否符合预期（裂隙应该是白色但不应过多）
        white_ratio = np.sum(self.binary_image > 0) / (self.binary_image.shape[0] * self.binary_image.shape[1])
        
        # 如果白色像素占比过高或过低，可能需要调整
        if white_ratio > 0.5:
            print(f"警告：二值化后白色像素占比过高 ({white_ratio:.2f})，使用了{binary_method}。可能需要调整阈值或反转图像")
            # 如果白色像素过多，反转图像
            self.binary_image = 255 - self.binary_image
            print("已自动反转二值图像")
        elif white_ratio < 0.01:
            print(f"警告：二值化后白色像素占比过低 ({white_ratio:.2f})，使用了{binary_method}。可能需要调整阈值或反转图像")
            # 如果白色像素太少，可能需要反转
            if avg_brightness > 100:  # 只在原图不是太暗的情况下反转
                self.binary_image = 255 - self.binary_image
                print("已自动反转二值图像")
        
        # 记录白色像素比例
        self.results['white_pixel_ratio'] = white_ratio
        
        return True
    
    def skeletonize(self):
        """提取裂隙骨架"""
        if self.binary_image is None:
            print("错误：请先进行图片预处理")
            return False
            
        # 骨架化
        self.skeleton = cv2.ximgproc.thinning(self.binary_image)
        
        return True
    
    def calculate_crack_length(self):
        """计算裂隙总长度"""
        if self.skeleton is None:
            print("错误：请先提取裂隙骨架")
            return 0
            
        # 使用连通分量分析来计算裂隙长度
        # 这种方法比简单计数像素更准确，特别是对于斜向裂隙
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(self.skeleton, connectivity=8)
        
        # 计算总长度
        total_length = 0
        
        # 跳过背景（标签0）
        for i in range(1, num_labels):
            # 获取当前连通分量的坐标
            y, x = np.where(labels == i)
            
            if len(x) <= 1:
                continue  # 跳过单个像素
                
            # 创建点集
            points = np.column_stack((x, y))
            
            # 计算当前裂隙段的长度
            segment_length = 0
            
            # 使用最小生成树方法计算长度
            # 首先构建邻接矩阵
            n_points = len(points)
            if n_points > 1:
                # 对于小段裂隙，直接计算欧氏距离
                if n_points <= 3:
                    for j in range(n_points-1):
                        dx = points[j+1, 0] - points[j, 0]
                        dy = points[j+1, 1] - points[j, 1]
                        segment_length += np.sqrt(dx*dx + dy*dy)
                else:
                    # 对于较长的裂隙，使用更复杂的方法
                    # 首先按照x坐标排序
                    sorted_indices = np.argsort(points[:, 0])
                    sorted_points = points[sorted_indices]
                    
                    # 计算相邻点之间的距离
                    for j in range(n_points-1):
                        dx = sorted_points[j+1, 0] - sorted_points[j, 0]
                        dy = sorted_points[j+1, 1] - sorted_points[j, 1]
                        segment_length += np.sqrt(dx*dx + dy*dy)
            
            total_length += segment_length
        
        # 如果连通分量分析失败或结果不合理，回退到简单像素计数
        if total_length == 0 or np.isnan(total_length):
            print("警告：连通分量分析计算裂隙长度失败，回退到像素计数方法")
            total_length = np.sum(self.skeleton > 0)
        
        # 保存结果
        self.results['crack_length_pixels'] = total_length
        
        return total_length
    
    def calculate_crack_width(self):
        """计算裂隙平均宽度"""
        if self.binary_image is None or self.skeleton is None:
            print("错误：请先进行图片预处理和骨架提取")
            return 0
            
        # 计算裂隙面积（二值图中白色像素数量）
        crack_area = np.sum(self.binary_image > 0)
        
        # 获取裂隙长度（如果尚未计算，则计算）
        if 'crack_length_pixels' not in self.results:
            crack_length = self.calculate_crack_length()
        else:
            crack_length = self.results['crack_length_pixels']
        
        if crack_length == 0:
            return 0
            
        # 平均宽度 = 面积 / 长度
        avg_width = crack_area / crack_length
        
        # 检查计算结果是否合理
        if avg_width > 50:  # 如果宽度过大，可能是计算有误
            print(f"警告：计算的平均宽度 ({avg_width:.2f}) 过大，可能不准确")
            
            # 尝试使用另一种方法计算
            # 使用距离变换计算平均宽度
            dist_transform = cv2.distanceTransform(self.binary_image, cv2.DIST_L2, 5)
            # 只考虑骨架上的点
            skeleton_points = self.skeleton > 0
            if np.sum(skeleton_points) > 0:
                # 计算骨架上点的平均距离变换值，并乘以2（半径到直径）
                alternative_width = np.mean(dist_transform[skeleton_points]) * 2
                print(f"使用距离变换计算的平均宽度: {alternative_width:.2f}")
                
                # 如果替代方法的结果更合理，使用它
                if alternative_width < avg_width and alternative_width > 0:
                    avg_width = alternative_width
                    print(f"使用距离变换计算的平均宽度作为结果: {avg_width:.2f}")
        
        self.results['crack_width_pixels'] = avg_width
        
        return avg_width
    
    def identify_branches(self):
        """识别裂隙分支"""
        if self.skeleton is None:
            print("错误：请先提取裂隙骨架")
            return []
            
        # 计算每个像素的邻居数量
        kernel = np.ones((3, 3), np.uint8)
        neighbors = cv2.filter2D(self.skeleton.astype(np.uint8), -1, kernel) - 1
        
        # 分支点是具有3个或更多邻居的点
        branch_points = np.logical_and(self.skeleton > 0, neighbors >= 3)
        branch_coords = np.column_stack(np.where(branch_points))
        
        self.results['branch_points'] = branch_coords
        self.results['branch_count'] = len(branch_coords)
        
        return branch_coords
    
    def calculate_crack_density(self):
        """计算裂隙密度（裂隙面积占总面积的比例）"""
        if self.binary_image is None:
            print("错误：请先进行图片预处理")
            return 0
            
        total_pixels = self.binary_image.shape[0] * self.binary_image.shape[1]
        crack_pixels = np.sum(self.binary_image > 0)
        
        # 基础裂隙密度计算
        basic_density = crack_pixels / total_pixels
        
        # 考虑裂隙分布的均匀性
        # 将图像分成网格，计算每个网格中的裂隙密度
        grid_size = 4  # 4x4网格
        height, width = self.binary_image.shape
        cell_height = height // grid_size
        cell_width = width // grid_size
        
        grid_densities = []
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算当前网格的边界
                top = i * cell_height
                bottom = min((i + 1) * cell_height, height)
                left = j * cell_width
                right = min((j + 1) * cell_width, width)
                
                # 提取当前网格
                cell = self.binary_image[top:bottom, left:right]
                
                # 计算当前网格的裂隙密度
                cell_total = cell.shape[0] * cell.shape[1]
                cell_cracks = np.sum(cell > 0)
                cell_density = cell_cracks / cell_total if cell_total > 0 else 0
                
                grid_densities.append(cell_density)
        
        # 计算网格密度的标准差，用于评估分布均匀性
        grid_std = np.std(grid_densities)
        
        # 计算非零网格的比例，用于评估裂隙覆盖范围
        non_zero_grids = sum(1 for d in grid_densities if d > 0.01)
        coverage_ratio = non_zero_grids / (grid_size * grid_size)
        
        # 综合考虑基础密度、分布均匀性和覆盖范围
        # 均匀性权重：0.2，覆盖范围权重：0.3，基础密度权重：0.5
        uniformity_factor = max(0, 1 - grid_std * 2)  # 标准差越小，均匀性越高
        
        density = (basic_density * 0.5 + 
                  uniformity_factor * 0.2 + 
                  coverage_ratio * 0.3)
        
        # 保存结果
        self.results['crack_density'] = density
        self.results['basic_density'] = basic_density
        self.results['uniformity_factor'] = uniformity_factor
        self.results['coverage_ratio'] = coverage_ratio
        
        return density
    
    def analyze_crack_orientation(self):
        """分析裂隙方向分布"""
        if self.skeleton is None:
            print("错误：请先提取裂隙骨架")
            return {}
            
        # 使用Hough变换检测线段
        lines = cv2.HoughLinesP(self.skeleton, 1, np.pi/180, 
                               threshold=10, minLineLength=10, maxLineGap=5)
        
        if lines is None:
            return {}
            
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # 垂直线
                angle = 90
            else:
                angle = np.abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
            angles.append(angle)
        
        # 计算角度分布
        angle_bins = np.zeros(18)  # 0-180度，每10度一个bin
        for angle in angles:
            bin_idx = min(17, int(angle / 10))
            angle_bins[bin_idx] += 1
            
        # 归一化
        if len(angles) > 0:
            angle_bins = angle_bins / len(angles)
            
        self.results['orientation_distribution'] = angle_bins
        
        # 计算主方向（出现最多的方向区间）
        if len(angles) > 0:
            main_direction = np.argmax(angle_bins) * 10 + 5  # bin中心值
            self.results['main_direction'] = main_direction
            
        # 保存裂隙条数（即检测到的线段数量）
        self.results['crack_count'] = len(lines) if lines is not None else 0
        
        # 保存裂隙走向分布
        self.results['crack_orientations'] = angles
            
        return angle_bins
    
    def calculate_rock_fragmentation(self):
        """计算岩体破碎程度
        
        岩体破碎程度基于裂隙密度、裂隙条数、分支点数量和裂隙长度综合评估
        """
        if 'crack_density' not in self.results or 'branch_count' not in self.results:
            print("错误：请先计算裂隙密度和分支点")
            return 0
        
        # 获取图像尺寸
        if self.image is None:
            return 0
        
        image_area = self.image.shape[0] * self.image.shape[1]
        
        # 计算裂隙密度因子 (0-1)
        density_factor = self.results['crack_density']
        
        # 计算分支复杂度因子 (0-1)
        # 使用更合理的归一化方法，基于图像大小动态调整
        image_size_factor = np.sqrt(image_area) / 1000  # 考虑图像尺寸的影响
        branch_density = self.results['branch_count'] / (20 * image_size_factor)  # 归一化
        branch_factor = min(1.0, branch_density)
        
        # 计算裂隙条数因子 (0-1)
        if 'crack_count' in self.results and self.results['crack_count'] > 0:
            crack_count = self.results['crack_count']
            # 动态调整最大裂隙条数，基于图像大小
            max_cracks = 50 * image_size_factor
            crack_count_factor = min(1.0, crack_count / max_cracks)
        else:
            crack_count_factor = 0
        
        # 计算裂隙长度因子 (0-1)
        if 'crack_length_pixels' in self.results and self.results['crack_length_pixels'] > 0:
            crack_length = self.results['crack_length_pixels']
            # 归一化裂隙长度，考虑图像周长
            image_perimeter = 2 * (self.image.shape[0] + self.image.shape[1])
            length_factor = min(1.0, crack_length / (2 * image_perimeter))
        else:
            length_factor = 0
        
        # 综合评估岩体破碎程度 (0-100)
        # 调整权重：裂隙密度(0.4)、分支复杂度(0.2)、裂隙条数(0.2)、裂隙长度(0.2)
        fragmentation_degree = (density_factor * 0.4 + 
                               branch_factor * 0.2 + 
                               crack_count_factor * 0.2 + 
                               length_factor * 0.2) * 100
        
        # 保存结果
        self.results['rock_fragmentation_degree'] = fragmentation_degree
        
        # 保存各个因子的值，便于分析
        self.results['density_factor'] = density_factor
        self.results['branch_factor'] = branch_factor
        self.results['crack_count_factor'] = crack_count_factor
        self.results['length_factor'] = length_factor
        
        return fragmentation_degree
    
    def perform_full_analysis(self, image_path, threshold_value=127, blur_size=5):
        """
        执行完整的裂隙分析
        
        参数:
            image_path: 图片路径
            threshold_value: 二值化阈值
            blur_size: 高斯模糊核大小
            
        返回:
            分析结果字典
        """
        if not self.load_image(image_path):
            return None
            
        if not self.preprocess_image(threshold_value, blur_size):
            return None
            
        if not self.skeletonize():
            return None
            
        # 执行各项分析
        self.calculate_crack_length()
        self.calculate_crack_width()
        self.identify_branches()
        self.calculate_crack_density()
        self.analyze_crack_orientation()
        self.calculate_rock_fragmentation()
        
        return self.results 