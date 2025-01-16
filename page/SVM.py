import os
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, \
    QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene
from PySide6.QtGui import QFont
from qfluentwidgets import Pivot, setTheme, Theme, SegmentedWidget, FluentIcon, PushButton
import algorithm.dataanalysisalgo as algo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# 支持向量机
class SVMPage(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout1.setObjectName(u"horizontalLayout1")
        self.horizontalLayout2 = QHBoxLayout()
        self.horizontalLayout2.setObjectName(u"horizontalLayout2")
        self.pagelabel = QLabel('支持向量机')
        self.font = QFont()
        self.font.setPointSize(28)
        self.pagelabel.setFont(self.font)
        self.verticalLayout.addWidget(self.pagelabel)

        self.verticalLayout.addLayout(self.horizontalLayout1)
        self.verticalLayout.addLayout(self.horizontalLayout2)

        self.textedit = QTextEdit(self)
        # self.textedit.setMinimumSize(0, 500)
        self.tableView = QTableView(self)
        self.graphicsView = QGraphicsView(self)

        self.openfilebtn = PushButton(self)
        self.openfilebtn.setText("打开文件目录")
        self.horizontalLayout1.addWidget(self.openfilebtn)

        self.readfilebtn = PushButton(self)
        self.readfilebtn.setText("读取数据文件")
        self.horizontalLayout1.addWidget(self.readfilebtn)
        self.filepathlabel_1 = QLabel(self)
        self.horizontalLayout1.addWidget(self.filepathlabel_1)  # 添加到布局中

        self.horizontalLayout2.addWidget(self.textedit)
        self.horizontalLayout2.addWidget(self.graphicsView)
        self.horizontalLayout2.addWidget(self.tableView)
