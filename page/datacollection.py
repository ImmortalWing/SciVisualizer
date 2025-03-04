import os
import cv2
import numpy as np
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtWidgets import (QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
                              QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene, QGridLayout, QGroupBox)
from PySide6.QtGui import QImage, QPixmap
from qfluentwidgets import (Pivot, setTheme, Theme, SegmentedWidget, FluentIcon, PushButton, InfoBar, InfoBarPosition, 
                           MessageBox, ImageLabel, Slider, SpinBox, ComboBox, LineEdit, CheckBox, StateToolTip)
import algorithm.dataanalysisalgo as algo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from algorithm.crack_quantification import CrackQuantification
import algorithm.crackidentificationalgo as crack_algo
import platform
import subprocess

# 极限学习机
class DataCollectionPage(QWidget):
    def __init__(self, text: str, parent=None):
        # 初始化裂隙量化分析器
        super().__init__(parent=parent)
        self.crack_analyzer = CrackQuantification()
        self.setObjectName(text.replace(' ', '-'))
        
        # 初始化输入图片路径变量
        self.input_image_path = None
        self.processed_image = None
        self.extracted_crack_path = None
        self.stateTooltip = None
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self)
        
        # 创建顶部控制区域
        self.control_layout = QHBoxLayout()
        
        # 添加读取图片按钮
        self.readfilebtn = PushButton("读取图片", self)
        self.readfilebtn.setIcon(FluentIcon.PHOTO)
        self.readfilebtn.clicked.connect(self.select_image)
        self.control_layout.addWidget(self.readfilebtn)
        
        # 添加裂隙提取按钮
        self.extract_crack_button = PushButton('裂隙提取', self)
        self.extract_crack_button.clicked.connect(self.extract_crack)
        self.control_layout.addWidget(self.extract_crack_button)
        
        # 添加裂隙量化按钮
        self.quantify_crack_button = PushButton('裂隙量化分析', self)
        self.quantify_crack_button.clicked.connect(self.perform_crack_quantification)
        self.control_layout.addWidget(self.quantify_crack_button)
        
        # 添加弹性空间
        self.control_layout.addStretch(1)
        
        # 将控制布局添加到主布局
        self.main_layout.addLayout(self.control_layout)
        
        # 创建中间区域（图片和参数设置）
        self.middle_layout = QHBoxLayout()
        
        # 创建图片显示区域
        self.image_layout = QVBoxLayout()
        
        # 添加图片标签
        self.readimagelabel = ImageLabel(self)
        self.readimagelabel.setFixedSize(600, 400)
        self.readimagelabel.setText("点击或拖拽图片到此处")
        self.readimagelabel.setAlignment(Qt.AlignCenter)
        self.readimagelabel.setScaledContents(False)  # 不自动缩放内容，使用我们的自定义缩放
        self.image_layout.addWidget(self.readimagelabel, 0, Qt.AlignCenter)
        
        # 添加结果图片标签
        self.result_image_label = ImageLabel(self)
        self.result_image_label.setFixedSize(600, 400)
        self.result_image_label.setText("分析结果将显示在这里")
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setScaledContents(False)  # 不自动缩放内容，使用我们的自定义缩放
        self.result_image_label.setVisible(False)  # 初始隐藏
        self.image_layout.addWidget(self.result_image_label, 0, Qt.AlignCenter)
        
        # 将图片布局添加到中间布局
        self.middle_layout.addLayout(self.image_layout)
        
        # 创建参数设置区域
        self.params_group = QGroupBox("分析参数设置", self)
        self.params_layout = QGridLayout()
        
        # 阈值设置
        self.threshold_label = QLabel("二值化阈值:", self)
        self.threshold_slider = Slider(Qt.Horizontal, self)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(120)
        self.threshold_value = SpinBox(self)
        self.threshold_value.setRange(0, 255)
        self.threshold_value.setValue(120)
        self.threshold_slider.valueChanged.connect(self.threshold_value.setValue)
        self.threshold_value.valueChanged.connect(self.threshold_slider.setValue)
        self.params_layout.addWidget(self.threshold_label, 0, 0)
        self.params_layout.addWidget(self.threshold_slider, 0, 1)
        self.params_layout.addWidget(self.threshold_value, 0, 2)
        
        # 模糊大小设置
        self.blur_label = QLabel("模糊大小:", self)
        self.blur_value = SpinBox(self)
        self.blur_value.setRange(1, 21)
        self.blur_value.setValue(5)
        self.blur_value.setSingleStep(2)  # 确保是奇数
        self.params_layout.addWidget(self.blur_label, 1, 0)
        self.params_layout.addWidget(self.blur_value, 1, 1, 1, 2)
        
        # 预览按钮
        self.preview_button = PushButton("预览处理效果", self)
        self.preview_button.clicked.connect(self.preview_processing)
        self.params_layout.addWidget(self.preview_button, 2, 0, 1, 3)
        
        # 高级选项
        self.advanced_options = QGroupBox("高级选项", self)
        self.advanced_layout = QVBoxLayout()
        
        # 是否应用形态学操作
        self.use_morphology = CheckBox("应用形态学操作", self)
        self.use_morphology.setChecked(True)
        self.advanced_layout.addWidget(self.use_morphology)
        
        # 是否显示中间结果
        self.show_intermediate = CheckBox("显示中间处理结果", self)
        self.show_intermediate.setChecked(False)
        self.advanced_layout.addWidget(self.show_intermediate)
        
        # 设置高级选项布局
        self.advanced_options.setLayout(self.advanced_layout)
        self.params_layout.addWidget(self.advanced_options, 3, 0, 1, 3)
        
        # 添加详细结果文本框
        self.results_text_group = QGroupBox("详细分析结果", self)
        self.results_text_layout = QVBoxLayout()
        
        self.results_text = QTextEdit(self)
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText("详细分析结果将显示在这里...")
        # 设置文本框样式
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #dcdcdc;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Microsoft YaHei', '微软雅黑', sans-serif;
                font-size: 12px;
                line-height: 1.5;
            }
        """)
        self.results_text_layout.addWidget(self.results_text)
        
        self.results_text_group.setLayout(self.results_text_layout)
        self.params_layout.addWidget(self.results_text_group, 4, 0, 1, 3)
        self.results_text_group.setVisible(False)  # 初始隐藏
        
        # 设置参数组布局
        self.params_group.setLayout(self.params_layout)
        self.middle_layout.addWidget(self.params_group)
        
        # 将中间布局添加到主布局
        self.main_layout.addLayout(self.middle_layout)
        
        # 添加结果显示区域
        self.results_group = QGroupBox("分析结果", self)
        self.results_layout = QGridLayout()
        
        # 结果标签
        self.crack_length_label = QLabel("裂隙长度: - 像素", self)
        self.crack_width_label = QLabel("平均宽度: - 像素", self)
        self.branch_count_label = QLabel("分支点数量: -", self)
        self.crack_density_label = QLabel("裂隙密度: - %", self)
        self.main_direction_label = QLabel("主要方向: - °", self)
        self.crack_count_label = QLabel("裂隙条数: -", self)
        self.rock_fragmentation_label = QLabel("岩体破碎程度: - %", self)
        
        self.results_layout.addWidget(self.crack_count_label, 0, 0)
        self.results_layout.addWidget(self.crack_length_label, 0, 1)
        self.results_layout.addWidget(self.crack_width_label, 1, 0)
        self.results_layout.addWidget(self.branch_count_label, 1, 1)
        self.results_layout.addWidget(self.crack_density_label, 2, 0)
        self.results_layout.addWidget(self.main_direction_label, 2, 1)
        self.results_layout.addWidget(self.rock_fragmentation_label, 3, 0, 1, 2)
        
        # 设置结果组布局
        self.results_group.setLayout(self.results_layout)
        self.results_group.setVisible(False)  # 初始隐藏
        self.main_layout.addWidget(self.results_group)
        
        # 添加弹性空间
        self.main_layout.addStretch(1)
        
        # 设置布局
        self.setLayout(self.main_layout)

    def preview_processing(self):
        """预览图像处理效果"""
        if not self.input_image_path:
            InfoBar.error(
                title='错误',
                content='请先选择一张图片',
                parent=self
            )
            return
            
        # 获取参数
        threshold_value = self.threshold_value.value()
        blur_size = self.blur_value.value()
        
        # 确保blur_size是奇数
        if blur_size % 2 == 0:
            blur_size += 1
            self.blur_value.setValue(blur_size)
        
        # 读取图像
        image = cv2.imread(self.input_image_path)
        if image is None:
            InfoBar.error(
                title='错误',
                content='无法读取图片',
                parent=self
            )
            return
            
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # 二值化
        _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
        
        # 如果选择了形态学操作
        if self.use_morphology.isChecked():
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 保存处理后的图像
        temp_path = os.path.join(os.path.dirname(self.input_image_path), "temp_preview.png")
        cv2.imwrite(temp_path, binary)
        
        # 显示处理后的图像
        self.result_image_label.setVisible(True)
        self.load_and_resize_image(temp_path, self.result_image_label)
        
        # 保存处理后的图像供后续分析
        self.processed_image = binary

    def extract_crack(self):
        """提取裂隙"""
        if not self.input_image_path:
            InfoBar.error(
                title='错误',
                content='请先选择一张图片',
                parent=self
            )
            return
            
        # 显示处理中状态
        if self.stateTooltip:
            self.stateTooltip.setState(True)
        else:
            self.stateTooltip = StateToolTip('正在处理', '正在提取裂隙...', self)
            self.stateTooltip.move(self.width()//2 - self.stateTooltip.width()//2,
                                  self.height()//2 - self.stateTooltip.height()//2)
            self.stateTooltip.show()
            
        try:
            # 调用裂隙提取算法
            self.extracted_crack_path = crack_algo.crackidentification(self.input_image_path)
            
            # 显示提取结果
            if os.path.exists(self.extracted_crack_path):
                self.result_image_label.setVisible(True)
                self.load_and_resize_image(self.extracted_crack_path, self.result_image_label)
                
                # 更新状态并关闭提示
                if self.stateTooltip:
                    self.stateTooltip.setContent("裂隙提取完成")
                    self.stateTooltip.setState(False)
                    self.stateTooltip = None
                
                InfoBar.success(
                    title='成功',
                    content='裂隙提取完成',
                    orient=Qt.Horizontal,
                    isClosable=True,
                    position=InfoBarPosition.TOP,
                    duration=2000,
                    parent=self
                )
        except Exception as e:
            # 处理错误
            if self.stateTooltip:
                self.stateTooltip.setState(False)
                self.stateTooltip = None
                
            InfoBar.error(
                title='错误',
                content=f'裂隙提取时出错: {str(e)}',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )

    def perform_crack_quantification(self):
        """执行裂隙量化分析"""
        # 获取当前选中的图片路径
        image_path = self.extracted_crack_path if self.extracted_crack_path else self.input_image_path
        
        if not image_path:
            InfoBar.error(
                title='错误',
                content='请先选择一张图片或提取裂隙',
                parent=self
            )
            return
        
        # 获取参数
        threshold_value = self.threshold_value.value()
        blur_size = self.blur_value.value()
        
        # 确保blur_size是奇数
        if blur_size % 2 == 0:
            blur_size += 1
            self.blur_value.setValue(blur_size)
        
        # 显示处理中状态
        if self.stateTooltip:
            self.stateTooltip.setState(True)
        else:
            self.stateTooltip = StateToolTip('正在处理', '正在进行裂隙量化分析...', self)
            self.stateTooltip.move(self.width()//2 - self.stateTooltip.width()//2,
                                  self.height()//2 - self.stateTooltip.height()//2)
            self.stateTooltip.show()
        
        try:
            # 执行裂隙分析
            results = self.crack_analyzer.perform_full_analysis(
                image_path=image_path,
                threshold_value=threshold_value,
                blur_size=blur_size
            )
            
            # 显示结果
            if results:
                self.show_quantification_results(results)
                
                # 如果选择了显示中间结果，可以在这里添加额外的处理
                if self.show_intermediate.isChecked():
                    # 生成可视化结果
                    output_path = os.path.join(os.path.dirname(image_path), "quantification_result.png")
                    self.crack_analyzer.visualize_results(output_path)
                    
                    # 显示可视化结果
                    if os.path.exists(output_path):
                        self.result_image_label.setVisible(True)
                        self.load_and_resize_image(output_path, self.result_image_label)
            else:
                InfoBar.error(
                    title='错误',
                    content='裂隙分析失败，未返回结果',
                    parent=self
                )
                
            # 更新状态并关闭提示 - 移到这里确保在所有处理完成后关闭
            if self.stateTooltip:
                self.stateTooltip.setContent("分析完成")
                self.stateTooltip.setState(False)
                self.stateTooltip = None
                
        except Exception as e:
            # 处理错误
            if self.stateTooltip:
                self.stateTooltip.setState(False)
                self.stateTooltip = None
                
            InfoBar.error(
                title='错误',
                content=f'裂隙量化分析时出错: {str(e)}',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=3000,
                parent=self
            )

    def show_quantification_results(self, results):
        """显示裂隙量化分析结果"""
        # 更新结果标签
        self.crack_length_label.setText(f"裂隙长度: {results.get('crack_length_pixels', 0):.1f} 像素")
        self.crack_width_label.setText(f"平均宽度: {results.get('crack_width_pixels', 0):.2f} 像素")
        self.branch_count_label.setText(f"分支点数量: {results.get('branch_count', 0)}")
        self.crack_density_label.setText(f"裂隙密度: {results.get('crack_density', 0)*100:.2f}%")
        self.crack_count_label.setText(f"裂隙条数: {results.get('crack_count', 0)}")
        
        if 'main_direction' in results:
            self.main_direction_label.setText(f"主要方向: {results['main_direction']:.1f}°")
        else:
            self.main_direction_label.setText("主要方向: - °")
            
        if 'rock_fragmentation_degree' in results:
            self.rock_fragmentation_label.setText(f"岩体破碎程度: {results['rock_fragmentation_degree']:.1f}%")
        else:
            self.rock_fragmentation_label.setText("岩体破碎程度: - %")
        
        # 生成详细结果文本
        detailed_text = "===== 裂隙量化分析详细结果 =====\n\n"
        detailed_text += f"裂隙条数: {results.get('crack_count', 0)}\n"
        detailed_text += f"裂隙总长度: {results.get('crack_length_pixels', 0):.1f} 像素\n"
        detailed_text += f"裂隙平均宽度: {results.get('crack_width_pixels', 0):.2f} 像素\n"
        detailed_text += f"分支点数量: {results.get('branch_count', 0)}\n"
        detailed_text += f"裂隙密度: {results.get('crack_density', 0)*100:.2f}%\n"
        
        if 'main_direction' in results:
            detailed_text += f"主要走向: {results['main_direction']:.1f}°\n"
            
        if 'rock_fragmentation_degree' in results:
            frag_degree = results['rock_fragmentation_degree']
            detailed_text += f"岩体破碎程度: {frag_degree:.1f}%\n"
            
            # 添加岩体破碎程度评估
            if frag_degree < 20:
                detailed_text += "岩体破碎程度评估: 完整性好，破碎程度低\n"
            elif frag_degree < 40:
                detailed_text += "岩体破碎程度评估: 轻度破碎\n"
            elif frag_degree < 60:
                detailed_text += "岩体破碎程度评估: 中度破碎\n"
            elif frag_degree < 80:
                detailed_text += "岩体破碎程度评估: 高度破碎\n"
            else:
                detailed_text += "岩体破碎程度评估: 极度破碎，完整性差\n"
                
        # 添加裂隙走向分布信息
        if 'orientation_distribution' in results:
            detailed_text += "\n裂隙走向分布:\n"
            dist = results['orientation_distribution']
            for i, value in enumerate(dist):
                angle_range = f"{i*10}-{(i+1)*10}"
                detailed_text += f"  {angle_range}°: {value*100:.1f}%\n"
                
        # 更新文本框
        self.results_text.setText(detailed_text)
        
        # 显示结果区域
        self.results_group.setVisible(True)
        self.results_text_group.setVisible(True)  # 显示详细结果文本框
        
        # 显示成功消息
        InfoBar.success(
            title='成功',
            content='裂隙量化分析完成',
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )

    def select_image(self):
        """选择图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.input_image_path = file_path
            
            # 限制图片大小
            self.load_and_resize_image(file_path, self.readimagelabel)
            
            # 重置提取的裂隙路径
            self.extracted_crack_path = None
            
            # 隐藏结果图片和结果区域
            self.result_image_label.setVisible(False)
            self.results_group.setVisible(False)
            self.results_text_group.setVisible(False)  # 隐藏详细结果文本框
            
            InfoBar.success(
                title='成功',
                content='图片加载成功',
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            
    def load_and_resize_image(self, image_path, target_label):
        """加载并调整图片大小"""
        if not os.path.exists(image_path):
            return
            
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            return
            
        # 获取标签尺寸
        label_width = target_label.width()
        label_height = target_label.height()
        
        # 获取图片尺寸
        img_height, img_width = image.shape[:2]
        
        # 计算缩放比例
        width_ratio = label_width / img_width
        height_ratio = label_height / img_height
        
        # 选择较小的比例以确保图片完全显示在标签内
        scale_ratio = min(width_ratio, height_ratio)
        
        # 如果图片比标签小，则不放大
        if scale_ratio > 1:
            scale_ratio = 1
            
        # 计算新尺寸
        new_width = int(img_width * scale_ratio)
        new_height = int(img_height * scale_ratio)
        
        # 调整图片大小
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # 转换为RGB（OpenCV使用BGR）
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # 创建QPixmap并设置到标签
        h, w, c = resized_image.shape
        qimg = QImage(resized_image.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # 设置到标签
        target_label.setPixmap(pixmap)
        
        # 居中显示
        target_label.setAlignment(Qt.AlignCenter)

    def showCommandBar(self):
        """显示命令栏"""
        pass
    
