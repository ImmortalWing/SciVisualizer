import os
from PySide6.QtCore import Qt, QAbstractTableModel, QSize
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, \
    QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene, QFrame
import algorithm.crackidentificationalgo as crack_algo
from PySide6.QtGui import QFont, QPixmap
from qfluentwidgets import Pivot, setTheme, Theme, SegmentedWidget, FluentIcon, PushButton, Action, ImageLabel, CommandBarView, FlyoutAnimationType, Flyout, \
    InfoBar, InfoBarPosition, StateToolTip, CardWidget, SubtitleLabel, BodyLabel, IconWidget, TransparentToolButton, \
    ToolTipPosition, ToolTip, SmoothScrollArea, TitleLabel, CaptionLabel, SwitchButton, ComboBox, IconInfoBadge, ToolTipFilter, InfoBadge
import algorithm.dataanalysisalgo as algo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class CrackIdentificationPage(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.stateTooltip = None
        self.input_image_path = ""
        self.output_image_path = ""
        
        # 设置主布局
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(30, 30, 30, 30)
        self.verticalLayout.setSpacing(20)
        
        # 页面标题区域
        self.setupHeaderArea()
        
        # 功能区域
        self.setupFunctionArea()
        
        # 图片显示区域
        self.setupImageArea()
        
        # 结果信息区域
        self.setupResultInfoArea()

    def setupHeaderArea(self):
        """设置页面标题区域"""
        self.headerCard = CardWidget(self)
        headerLayout = QHBoxLayout(self.headerCard)
        headerLayout.setContentsMargins(20, 15, 20, 15)
        
        # 标题图标
        self.titleIcon = IconWidget(FluentIcon.SEARCH, self.headerCard)
        self.titleIcon.setFixedSize(48, 48)
        headerLayout.addWidget(self.titleIcon)
        
        # 标题文本区域
        titleTextLayout = QVBoxLayout()
        titleTextLayout.setSpacing(0)
        
        self.titleLabel = TitleLabel('裂隙识别', self.headerCard)
        titleTextLayout.addWidget(self.titleLabel)
        
        self.subtitleLabel = BodyLabel('上传图片并进行裂隙自动识别分析', self.headerCard)
        titleTextLayout.addWidget(self.subtitleLabel)
        
        headerLayout.addLayout(titleTextLayout)
        headerLayout.addStretch(1)
        
        # 添加到主布局
        self.verticalLayout.addWidget(self.headerCard)

    def setupFunctionArea(self):
        """设置功能按钮区域"""
        self.functionCard = CardWidget(self)
        functionLayout = QVBoxLayout(self.functionCard)
        functionLayout.setContentsMargins(20, 15, 20, 15)
        functionLayout.setSpacing(15)
        
        # 功能区标题
        functionTitleLayout = QHBoxLayout()
        self.functionTitle = SubtitleLabel("操作面板", self.functionCard)
        functionTitleLayout.addWidget(self.functionTitle)
        functionTitleLayout.addStretch(1)
        functionLayout.addLayout(functionTitleLayout)
        
        # 分隔线
        self.separator = QFrame(self.functionCard)
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        functionLayout.addWidget(self.separator)
        
        # 功能按钮区域
        buttonsLayout = QHBoxLayout()
        buttonsLayout.setSpacing(15)
        
        # 读取图片按钮
        self.readfilebtn = PushButton(self.functionCard)
        self.readfilebtn.setText("读取图片")
        self.readfilebtn.setIcon(FluentIcon.PHOTO)
        self.readfilebtn.clicked.connect(self.select_image)
        # 修复工具提示
        self.readfilebtn.setToolTip("选择要分析的图片文件")
        self.readfilebtn.installEventFilter(ToolTipFilter(self.readfilebtn, showDelay=300, position=ToolTipPosition.BOTTOM))
        buttonsLayout.addWidget(self.readfilebtn)

        # 处理图片按钮
        self.switchBtn = PushButton(self.functionCard)
        self.switchBtn.setText("开始识别")
        self.switchBtn.setIcon(FluentIcon.PLAY)
        self.switchBtn.clicked.connect(self.switch_image)
        # 修复工具提示
        self.switchBtn.setToolTip("开始裂隙识别分析")
        self.switchBtn.installEventFilter(ToolTipFilter(self.switchBtn, showDelay=300, position=ToolTipPosition.BOTTOM))
        buttonsLayout.addWidget(self.switchBtn)
        
        # 保存结果按钮
        self.saveBtn = PushButton(self.functionCard)
        self.saveBtn.setText("保存结果")
        self.saveBtn.setIcon(FluentIcon.SAVE)
        self.saveBtn.clicked.connect(self.save_result)
        # 修复工具提示
        self.saveBtn.setToolTip("保存分析结果图片")
        self.saveBtn.installEventFilter(ToolTipFilter(self.saveBtn, showDelay=300, position=ToolTipPosition.BOTTOM))
        buttonsLayout.addWidget(self.saveBtn)
        
        # 文件路径显示
        self.filePathLayout = QHBoxLayout()
        self.filePathLabel = CaptionLabel("当前文件:", self.functionCard)
        self.filepathlabel_1 = BodyLabel("未选择文件", self.functionCard)
        self.filePathLayout.addWidget(self.filePathLabel)
        self.filePathLayout.addWidget(self.filepathlabel_1, 1)
        
        functionLayout.addLayout(buttonsLayout)
        functionLayout.addLayout(self.filePathLayout)
        
        # 添加到主布局
        self.verticalLayout.addWidget(self.functionCard)

    def setupImageArea(self):
        """设置图片显示区域"""
        self.imageAreaLayout = QHBoxLayout()
        self.imageAreaLayout.setSpacing(15)
        
        # 原始图片卡片
        self.originalImageCard = CardWidget(self)
        originalImageLayout = QVBoxLayout(self.originalImageCard)
        originalImageLayout.setContentsMargins(15, 15, 15, 15)
        
        # 原始图片标题
        originalTitleLayout = QHBoxLayout()
        self.originalTitle = SubtitleLabel("原始图片", self.originalImageCard)
        originalTitleLayout.addWidget(self.originalTitle)
        
        # 添加图片状态标签 - 修复：使用正确的静态方法创建
        self.originalImageBadge = InfoBadge("等待上传", self.originalImageCard)
        originalTitleLayout.addWidget(self.originalImageBadge, 0, Qt.AlignRight)
        originalTitleLayout.addStretch(1)
        originalImageLayout.addLayout(originalTitleLayout)
        
        # 原始图片显示
        self.readimagelabel = ImageLabel(self.originalImageCard)
        self.readimagelabel.setFixedSize(600, 400)
        self.readimagelabel.setAlignment(Qt.AlignCenter)
        self.readimagelabel.setScaledContents(False)  # 不自动缩放内容，使用我们的自定义缩放
        self.readimagelabel.clicked.connect(self.showCommandBar)
        originalImageLayout.addWidget(self.readimagelabel)
        
        # 结果图片卡片
        self.resultImageCard = CardWidget(self)
        resultImageLayout = QVBoxLayout(self.resultImageCard)
        resultImageLayout.setContentsMargins(15, 15, 15, 15)
        
        # 结果图片标题
        resultTitleLayout = QHBoxLayout()
        self.resultTitle = SubtitleLabel("识别结果", self.resultImageCard)
        resultTitleLayout.addWidget(self.resultTitle)
        
        # 添加结果状态标签 - 修复：使用正确的静态方法创建
        self.resultImageBadge = InfoBadge("等待处理", self.resultImageCard)
        resultTitleLayout.addWidget(self.resultImageBadge, 0, Qt.AlignRight)
        resultTitleLayout.addStretch(1)
        resultImageLayout.addLayout(resultTitleLayout)
        
        # 结果图片显示
        self.resultImageLabel = ImageLabel(self.resultImageCard)
        self.resultImageLabel.setFixedSize(600, 400)
        self.resultImageLabel.setAlignment(Qt.AlignCenter)
        self.resultImageLabel.setScaledContents(False)  # 不自动缩放内容，使用我们的自定义缩放
        self.resultImageLabel.clicked.connect(self.showCommandBar)
        resultImageLayout.addWidget(self.resultImageLabel)
        
        # 添加到水平布局
        self.imageAreaLayout.addWidget(self.originalImageCard, 1)
        self.imageAreaLayout.addWidget(self.resultImageCard, 1)
        
        # 添加到主布局
        self.verticalLayout.addLayout(self.imageAreaLayout)

    def setupResultInfoArea(self):
        """设置结果信息区域"""
        self.resultInfoCard = CardWidget(self)
        resultInfoLayout = QVBoxLayout(self.resultInfoCard)
        resultInfoLayout.setContentsMargins(20, 15, 20, 15)
        
        # 结果信息标题
        resultInfoTitleLayout = QHBoxLayout()
        self.resultInfoTitle = SubtitleLabel("分析结果", self.resultInfoCard)
        resultInfoTitleLayout.addWidget(self.resultInfoTitle)
        resultInfoTitleLayout.addStretch(1)
        resultInfoLayout.addLayout(resultInfoTitleLayout)
        
        # 分隔线
        self.resultSeparator = QFrame(self.resultInfoCard)
        self.resultSeparator.setFrameShape(QFrame.HLine)
        self.resultSeparator.setFrameShadow(QFrame.Sunken)
        resultInfoLayout.addWidget(self.resultSeparator)
        
        # 结果信息内容
        self.resultInfoContent = BodyLabel("请先上传图片并进行裂隙识别分析", self.resultInfoCard)
        resultInfoLayout.addWidget(self.resultInfoContent)
        
        # 添加到主布局
        self.verticalLayout.addWidget(self.resultInfoCard)

    def scale_image_for_display(self, image_path, label):
        """缩放图片以适应显示区域
        
        Args:
            image_path: 图片文件路径
            label: 要显示图片的ImageLabel控件
        """
        if not os.path.exists(image_path):
            return
            
        # 获取标签的尺寸
        label_width = label.width()
        label_height = label.height()
        
        # 设置最大显示尺寸限制
        max_width = 600  # 最大宽度限制
        max_height = 400  # 最大高度限制
        
        # 使用标签尺寸和最大限制中的较小值
        display_width = min(label_width, max_width)
        display_height = min(label_height, max_height)
        
        # 加载图片
        pixmap = QPixmap(image_path)
        original_width = pixmap.width()
        original_height = pixmap.height()
        
        # 计算缩放比例，保持宽高比
        width_ratio = display_width / original_width
        height_ratio = display_height / original_height
        
        # 使用较小的比例以确保图片完全显示在标签内
        scale_ratio = min(width_ratio, height_ratio)
        
        # 如果图片比标签小，则不放大（最大缩放比例为1）
        if scale_ratio > 1:
            scale_ratio = 1
            
        # 计算缩放后的尺寸
        new_width = int(original_width * scale_ratio * 0.9)  # 留出10%的边距
        new_height = int(original_height * scale_ratio * 0.9)  # 留出10%的边距
        
        # 缩放图片并设置到标签
        scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)
        
        # 保存原始图片路径到标签的属性中，以便窗口大小变化时可以重新缩放
        label.setProperty("original_image_path", image_path)
        
        # 返回缩放后的图片，以便可能的进一步处理
        return scaled_pixmap

    def select_image(self):
        """选择图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "resource", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.input_image_path = file_path
            # 使用自定义缩放方法替代直接设置图片
            self.scale_image_for_display(file_path, self.readimagelabel)
            self.filepathlabel_1.setText(os.path.basename(file_path))
            
            # 更新图片状态标签
            self.originalImageBadge.setText("已上传")
            
            self.show_info_bar('成功', '图片加载成功', 'success')

    def switch_image(self):
        """处理并显示结果图"""
        if not self.input_image_path:
            self.show_info_bar('警告', '请先选择图片', 'warning')
            return

        # 更新结果状态标签
        self.resultImageBadge.setText("处理中...")

        # 显示处理中状态
        if self.stateTooltip:
            self.stateTooltip.setState(True)
        else:
            self.stateTooltip = StateToolTip('正在处理', '请稍候...', self)
            self.stateTooltip.move(self.width()//2 - self.stateTooltip.width()//2,
                                  self.height()//2 - self.stateTooltip.height()//2)
            self.stateTooltip.show()

        try:
            # 调用算法处理
            output_path = crack_algo.crackidentification(self.input_image_path)

            # 显示处理结果
            if os.path.exists(output_path):
                # 使用自定义缩放方法替代直接设置图片
                self.scale_image_for_display(output_path, self.resultImageLabel)
                self.output_image_path = output_path  # 保存输出图片路径
                
                # 更新结果状态标签
                self.resultImageBadge.setText("已完成")
                
                # 更新结果信息
                self.resultInfoContent.setText(f"裂隙识别分析已完成。\n识别到的裂隙数量: {self.get_crack_count()}\n裂隙总长度: {self.get_crack_length()}")
                
                # 更新状态并关闭提示
                if self.stateTooltip:
                    self.stateTooltip.setContent("处理完成")
                    self.stateTooltip.setState(False)
                    self.stateTooltip = None
                
                self.show_info_bar('成功', '图片处理完成', 'success')
        except Exception as e:
            # 处理错误
            if self.stateTooltip:
                self.stateTooltip.setState(False)
                self.stateTooltip = None
            
            # 更新结果状态标签
            self.resultImageBadge.setText("处理失败")
                
            self.show_info_bar('错误', f'处理图片时出错: {str(e)}', 'error', 3000)

    def get_crack_count(self):
        """获取裂隙数量（示例方法，实际应从算法结果中获取）"""
        # 这里应该从算法结果中获取实际的裂隙数量
        return "5个"  # 示例数据
        
    def get_crack_length(self):
        """获取裂隙总长度（示例方法，实际应从算法结果中获取）"""
        # 这里应该从算法结果中获取实际的裂隙长度
        return "约125.6像素"  # 示例数据

    def save_result(self):
        """保存结果图片"""
        if not hasattr(self, 'output_image_path') or not self.output_image_path or not os.path.exists(self.output_image_path):
            self.show_info_bar('警告', '没有可保存的结果图片', 'warning')
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果图片", os.path.dirname(self.input_image_path), 
            "Image Files (*.png *.jpg *.jpeg)")
            
        if save_path:
            try:
                # 复制结果图片到选择的位置
                import shutil
                shutil.copy2(self.output_image_path, save_path)
                
                self.show_info_bar('成功', f'结果已保存至: {save_path}', 'success')
            except Exception as e:
                self.show_info_bar('错误', f'保存图片失败: {str(e)}', 'error')

    def showCommandBar(self, sender=None):
        view = CommandBarView(self)

        view.addAction(Action(FluentIcon.SHARE, '分享'))
        view.addAction(Action(FluentIcon.SAVE, '保存', triggered=self.save_result))
        view.addAction(Action(FluentIcon.DELETE, '删除'))

        view.addHiddenAction(Action(FluentIcon.APPLICATION, '应用', shortcut='Ctrl+A'))
        view.addHiddenAction(Action(FluentIcon.SETTING, '设置', shortcut='Ctrl+S'))
        view.resizeToSuitableWidth()

        # 如果没有指定sender，使用默认的readimagelabel
        if not sender:
            sender = self.readimagelabel
            
        Flyout.make(view, sender, self, FlyoutAnimationType.FADE_IN)

    def resizeEvent(self, event):
        """窗口大小变化时重新缩放图片"""
        super().resizeEvent(event)
        
        # 重新缩放原始图片
        if hasattr(self, 'readimagelabel') and self.readimagelabel.property("original_image_path"):
            self.scale_image_for_display(self.readimagelabel.property("original_image_path"), self.readimagelabel)
            
        # 重新缩放结果图片
        if hasattr(self, 'resultImageLabel') and self.resultImageLabel.property("original_image_path"):
            self.scale_image_for_display(self.resultImageLabel.property("original_image_path"), self.resultImageLabel)


    def show_info_bar(self, title, content, info_type='info', duration=2000):
        """显示通知栏
        
        参数:
            title (str): 标题
            content (str): 内容
            info_type (str): 类型，可选 'info', 'success', 'warning', 'error'
            duration (int): 显示时间（毫秒）
        """
        bar_func = {
            'info': InfoBar.info,
            'success': InfoBar.success,
            'warning': InfoBar.warning,
            'error': InfoBar.error
        }.get(info_type, InfoBar.info)
        
        bar_func(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=duration,
            parent=self
        )