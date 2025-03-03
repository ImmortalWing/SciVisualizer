<<<<<<< HEAD
from PySide6.QtWidgets import QApplication, QFrame, QStackedWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QRect, QUrl


class MachineLearningPage(QFrame):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = QLabel(text, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignCenter)
        self.setObjectName(text.replace(' ', '-'))

=======
from PySide6.QtWidgets import (QFrame, QLabel, QVBoxLayout, QHBoxLayout,
                              QWidget, QSpacerItem, QSizePolicy)
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
    
    /* 机器学习介绍样式 */
    .feature-title {
        font: bold 16px 'Microsoft YaHei';
        color: #2c3e50;
        margin: 15px 0;
        border-bottom: 2px solid #3a7afe;
        padding-bottom: 5px;
    }
    
    .feature-item {
        font-size: 13px;
        color: #4a5568;
        margin: 8px 0;
        padding-left: 10px;
    }
    
    IconWidget {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 5px;
    }
"""

class MachineLearningPage(QFrame):
    """机器学习主界面"""
    def __init__(self, text: str,parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.setup_ui()
        self.setStyleSheet(STYLESHEET)
        
    def setup_ui(self):
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(25)
        
        # 顶部区域
        self.create_header(main_layout)
        
        # 功能区域
        self.create_features(main_layout)
        
        # 介绍区域
        self.create_intro(main_layout)
        
        # 添加伸缩空间
        main_layout.addStretch(1)
        
    def create_header(self, layout):
        """创建头部区域（同home.py）"""
        header = QWidget()
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)
        
        # Logo
        logo_container = QWidget()
        logo_container.setStyleSheet(".logo-container { background-color: #3498db; }")
        logo_layout = QHBoxLayout(logo_container)
        logo = QLabel()
        pixmap = QPixmap("resource/logo.png")
        if not pixmap.isNull():
            logo.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio))
        else:
            logo.setText("SciVisualizer")
        logo_layout.addWidget(logo)
        
        # 标题
        title = QLabel("机器学习分析平台")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignCenter)
        
        h_layout.addWidget(logo_container)
        h_layout.addWidget(title, 1)
        layout.addWidget(header)
        
    def create_features(self, layout):
        """创建功能卡片（同home.py结构）"""
        from qfluentwidgets import FlowLayout
        
        features = [
            {"title": "模型训练", "desc": "支持多种机器学习算法配置"},
            {"title": "特征工程", "desc": "自动化特征选择与处理"},
            {"title": "模型评估", "desc": "可视化评估指标分析"},
            {"title": "预测分析", "desc": "实时数据预测与可视化"}
        ]
        
        container = QWidget()
        flow_layout = FlowLayout(container, needAni=False)
        flow_layout.setContentsMargins(0, 0, 0, 0)
        flow_layout.setHorizontalSpacing(12)
        flow_layout.setVerticalSpacing(12)
        
        routes = ["ModelTrain-Interface", "FeatureEngineering-Interface", 
                "ModelEval-Interface", "Prediction-Interface"]
        for idx, feature in enumerate(features):
            card = CardWidget(container)
            card.setFixedSize(360, 90)
            # 卡片内容设置（同home.py）
            
        layout.addWidget(container)
        
    def create_intro(self, layout):
        """创建机器学习介绍区域"""
        intro_widget = QWidget()
        h_layout = QHBoxLayout(intro_widget)
        h_layout.setContentsMargins(20, 30, 20, 30)
        h_layout.setSpacing(40)

        # 算法介绍列
        algo_col = QVBoxLayout()
        title1 = QLabel("核心功能")
        title1.setProperty("class", "feature-title")
        
        features = [
            "▪ 支持监督/非监督/强化学习",
            "▪ 自动化超参数调优",
            "▪ 特征重要性可视化分析",
            "▪ 模型可解释性（SHAP值）",
            "▪ 实时训练过程监控"
        ]
        for text in features:
            label = QLabel(text)
            label.setProperty("class", "feature-item")
            algo_col.addWidget(label)

        # 框架支持列
        framework_col = QVBoxLayout()
        title2 = QLabel("支持框架")
        title2.setProperty("class", "feature-title")
        
        icon_container = QWidget()
        flow_layout = FlowLayout(icon_container)
        frameworks = [
            ("sklearn", "Scikit-learn"),
            ("pytorch", "PyTorch"),
            ("tensorflow", "TensorFlow"),
            ("xgboost", "XGBoost")
        ]
        for icon, name in frameworks:
            wrapper = QWidget()
            vbox = QVBoxLayout(wrapper)
            icon_widget = IconWidget(f"resource/{icon}.png")
            icon_widget.setFixedSize(48, 48)
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            vbox.addWidget(icon_widget, 0, Qt.AlignCenter)
            vbox.addWidget(label)
            flow_layout.addWidget(wrapper)

        framework_col.addWidget(title2)
        framework_col.addWidget(icon_container)

        h_layout.addLayout(algo_col)
        h_layout.addLayout(framework_col)
        layout.addWidget(intro_widget)
>>>>>>> 5e7640a (version-1.1)
