from PySide6.QtWidgets import (QFrame, QLabel, QVBoxLayout, QHBoxLayout, 
                              QWidget, QSpacerItem, QSizePolicy, QGridLayout)
from qfluentwidgets import IconWidget, TextWrap, FlowLayout, CardWidget
from .common.signal_bus import signalBus

from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt, QSize

STYLESHEET = """
    /* 主容器样式 - 现代渐变背景 */
    HomePage {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 #f8f9fa, stop:1 #ffffff);
        padding: 40px;
        border: none;
    }
    
    /* 标题样式 - 增加字距 */
    .title {
        font: bold 28px 'Microsoft YaHei';
        color: #2c3e50;
        margin: 25px 0;
        letter-spacing: 2px;
    }
    
    /* 底部信息样式 */
    .footer-text {
        color: #a0aec0;
        font-size: 13px;
        margin-top: 40px;
        letter-spacing: 1px;
    }
    
    /* Logo容器样式 */
    .logo-container {
        background: qradialgradient(
            cx:0.5, cy:0.5, radius: 0.5,
            fx:0.5, fy:0.5,
            stop:0 #3a7afe33, stop:1 #3a7afe11
        );
        border-radius: 20px;
        padding: 15px;
    }
"""

class SampleCard(CardWidget):
    """ Sample card """

    def __init__(self, icon, title, content, routeKey, index, parent=None):
        super().__init__(parent=parent)
        #self.shadowEnabled = False  # 直接禁用阴影
        self.index = index
        self.routekey = routeKey

        self.iconWidget = IconWidget(icon, self)
        self.titleLabel = QLabel(title, self)
        self.contentLabel = QLabel(TextWrap.wrap(content, 45, False)[0], self)

        self.hBoxLayout = QHBoxLayout(self)
        self.vBoxLayout = QVBoxLayout()

        self.setFixedSize(360, 90)
        self.iconWidget.setFixedSize(48, 48)

        self.hBoxLayout.setSpacing(28)
        self.hBoxLayout.setContentsMargins(20, 0, 0, 0)
        self.vBoxLayout.setSpacing(2)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.setAlignment(Qt.AlignVCenter)

        self.hBoxLayout.setAlignment(Qt.AlignVCenter)
        self.hBoxLayout.addWidget(self.iconWidget)
        self.hBoxLayout.addLayout(self.vBoxLayout)
        self.vBoxLayout.addStretch(1)
        self.vBoxLayout.addWidget(self.titleLabel)
        self.vBoxLayout.addWidget(self.contentLabel)
        self.vBoxLayout.addStretch(1)

        self.titleLabel.setObjectName('titleLabel')
        self.contentLabel.setObjectName('contentLabel')

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        signalBus.switchToSampleCard.emit(self.routekey, self.index)


class FeatureCard(SampleCard):
    """集成Gallery样式的功能卡片"""
    def __init__(self, title: str, description: str, routeKey: str, index: int):
        super().__init__(
            icon="resource/kunkun.png",
            title=title,
            content=description,
            routeKey=routeKey,
            index=index,
            parent=None
        )
        self.setFixedSize(450, 100)
        self.iconWidget.setFixedSize(48, 48)

class HomePage(QFrame):
    """主首页组件"""
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setup_ui()
        self.setStyleSheet(STYLESHEET)
        self.setObjectName("homePage")  # 添加对象名便于样式和调试
        
    def setup_ui(self):
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(25)
        
        # 顶部区域
        self.create_header(main_layout)
        
        # 功能区域
        self.create_features(main_layout)
        
        # 底部区域
        self.create_footer(main_layout)
        
        # 添加伸缩空间
        main_layout.addStretch(1)
        
    def create_header(self, layout):
        """创建头部区域"""
        header = QWidget()
        header.setObjectName("headerWidget")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)
        
        # Logo
        logo_container = QWidget()
        logo_container.setObjectName("logoContainer")
        logo_container.setProperty("class", "logo-container")  # 使用样式表中的样式而不是重新定义
        logo_layout = QHBoxLayout(logo_container)
        logo = QLabel()
        logo.setObjectName("logoLabel")
        pixmap = QPixmap("resource/logo.png")
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
        else:
            logo.setText("SciVisualizer")
        logo_layout.addWidget(logo)
        
        # 标题
        title = QLabel("科学数据可视化分析平台")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignCenter)
        
        h_layout.addWidget(logo_container)
        h_layout.addWidget(title, 1)
        layout.addWidget(header)
        
    def create_features(self, layout):
        """创建网格布局功能区域"""
        features = [
            {"title": "数据可视化", "desc": "支持折线图、热力图、3D曲面等15+图表类型"},
            {"title": "机器学习", "desc": "集成分类、回归、聚类等经典算法"},
            {"title": "深度学习", "desc": "神经网络训练与可视化分析"},
            {"title": "数据分析", "desc": "数据清洗、统计建模与报告生成"}
        ]
        
        # 流式布局容器 - 使用顶部导入的FlowLayout
        container = QWidget()
        container.setObjectName("featureContainer")
        flow_layout = FlowLayout(container, needAni=False)
        flow_layout.setContentsMargins(0, 0, 0, 0)
        flow_layout.setHorizontalSpacing(12)
        flow_layout.setVerticalSpacing(12)
        
        # 创建并放置卡片
        routes = ["DataVisualizer-Interface", "Machinelearning-Interface", "Deeplearning-Interface", "DataAnalysis-Interface"]
        for idx, feature in enumerate(features):
            card = FeatureCard(
                title=feature["title"],
                description=feature["desc"],
                routeKey=routes[idx],
                index=idx
            )
            flow_layout.addWidget(card)
            
        layout.addWidget(container, stretch=0)
        
    def create_footer(self, layout):
        """创建底部区域"""
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(0, 20, 0, 0)
        
        # 版权信息
        copyright = QLabel("© 2025 SciVisualizer | MIT License")
        copyright.setAlignment(Qt.AlignCenter)
        copyright.setProperty("class", "footer-text")
        
        # 版本信息
        version = QLabel("Version 2.1.0 | Build 20250228")
        version.setAlignment(Qt.AlignCenter)
        version.setProperty("class", "footer-text")
        
        footer_layout.addWidget(copyright)
        footer_layout.addWidget(version)
        layout.addWidget(footer)
