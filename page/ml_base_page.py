import os
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, \
    QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene
from PySide6.QtGui import QFont
from qfluentwidgets import PushButton
import algorithm.dataanalysisalgo as algo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

class MLBasePage(QWidget):
    """机器学习算法页面的基类，提供共同的UI和功能"""
    
    def __init__(self, text: str, title: str, parent=None):
        """
        初始化机器学习基础页面
        
        Args:
            text: 对象名称的基础
            title: 页面标题
            parent: 父部件
        """
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.title = title
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI布局"""
        # 设置布局
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName(u"verticalLayout")
        
        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout1.setObjectName(u"horizontalLayout1")
        
        self.horizontalLayout2 = QHBoxLayout()
        self.horizontalLayout2.setObjectName(u"horizontalLayout2")
        
        # 创建标题标签
        self.pagelabel = QLabel(self.title)
        self.font = QFont()
        self.font.setPointSize(28)
        self.pagelabel.setFont(self.font)
        self.verticalLayout.addWidget(self.pagelabel)
        
        # 添加布局
        self.verticalLayout.addLayout(self.horizontalLayout1)
        self.verticalLayout.addLayout(self.horizontalLayout2)
        
        # 创建组件
        self.textedit = QTextEdit(self)
        self.tableView = QTableView(self)
        self.graphicsView = QGraphicsView(self)
        
        # 创建按钮
        self.openfilebtn = PushButton(self)
        self.openfilebtn.setText("打开文件目录")
        self.horizontalLayout1.addWidget(self.openfilebtn)
        
        self.readfilebtn = PushButton(self)
        self.readfilebtn.setText("读取数据文件")
        self.horizontalLayout1.addWidget(self.readfilebtn)
        
        self.filepathlabel_1 = QLabel(self)
        self.horizontalLayout1.addWidget(self.filepathlabel_1)
        
        # 添加到布局
        self.horizontalLayout2.addWidget(self.textedit)
        self.horizontalLayout2.addWidget(self.graphicsView)
        self.horizontalLayout2.addWidget(self.tableView)
        
        # 连接信号槽
        self._connect_signals()
        
    def _connect_signals(self):
        """连接信号和槽"""
        self.openfilebtn.clicked.connect(self.open_file_directory)
        self.readfilebtn.clicked.connect(self.read_data_file)
        
    def open_file_directory(self):
        """打开文件目录"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if folder_path:
            self.filepathlabel_1.setText(f"已选择目录: {os.path.basename(folder_path)}")
            self.current_directory = folder_path
            
    def read_data_file(self):
        """读取数据文件，子类需要实现此方法"""
        if hasattr(self, 'current_directory'):
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "选择数据文件", 
                self.current_directory, 
                "所有文件 (*);;CSV文件 (*.csv);;Excel文件 (*.xlsx *.xls)"
            )
            if file_path:
                self.process_data_file(file_path)
                
    def process_data_file(self, file_path):
        """处理数据文件，子类需要重写此方法"""
        self.textedit.append(f"读取文件: {file_path}")
        self.textedit.append("子类应实现数据处理逻辑")  # 子类需覆盖此方法