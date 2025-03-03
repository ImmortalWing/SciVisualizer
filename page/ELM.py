import os
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, \
    QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene
from PySide6.QtGui import QFont
<<<<<<< HEAD
from qfluentwidgets import Pivot, setTheme, Theme, SegmentedWidget, FluentIcon, PushButton
import algorithm.dataanalysisalgo as algo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


# 极限学习机
class ELMPage(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout1.setObjectName(u"horizontalLayout1")
        self.horizontalLayout2 = QHBoxLayout()
        self.horizontalLayout2.setObjectName(u"horizontalLayout2")
        self.pagelabel = QLabel('极限学习机')
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
=======
from qfluentwidgets import PushButton, FluentIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from algorithm import elm_algo as elm
import pandas as pd
from PySide6.QtCore import Signal, QRect
from PySide6.QtWidgets import QHeaderView, QSplitter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


# 极限学习机
class ELMModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

class ELMPage(QWidget):
    dataLoaded = Signal(pd.DataFrame)
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self._init_ui()

    def _init_ui(self):
        """优化后的现代化界面布局"""
        # 主分割器（左右布局）
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(10)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #e0e0e0;
                margin: 4px;
                border-radius: 4px;
            }
            QSplitter::handle:hover {
                background: #90a4ae;
            }
        """)

        # 左侧控制面板（30%宽度）
        control_panel = QWidget()
        control_panel.setMinimumWidth(280)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(12, 20, 12, 20)
        control_layout.setSpacing(15)

        # 标题区域
        title = QLabel('极限学习机分析平台')
        title.setFont(QFont('Microsoft YaHei', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            color: #2c3e50;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
        """)

        # 文件操作区域
        file_group = QWidget()
        file_layout = QHBoxLayout(file_group)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(8)

        self.open_btn = PushButton(FluentIcon.FOLDER, "选择文件", self)
        self.load_btn = PushButton(FluentIcon.DOWNLOAD, "加载数据", self)
        self.filepathlabel_1 = QLabel("当前文件：未选择")
        self.filepathlabel_1.setStyleSheet("""
            color: #666;
            font: 13px 'Microsoft YaHei';
        """)

        # 信息显示区域
        self.info_box = QTextEdit()
        self.info_box.setStyleSheet("""
            QTextEdit {
                background: #fff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
                padding: 8px;
                font: 13px Consolas;
            }
        """)

        # 右侧可视化区域（70%宽度）
        vis_splitter = QSplitter(Qt.Vertical)
        vis_splitter.setChildrenCollapsible(False)

        # 数据表格
        self.data_table = QTableView()
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.verticalHeader().setDefaultSectionSize(32)
        self.data_table.setStyleSheet("""
            QTableView {
                background: #fff;
                border: 1px solid #dee2e6;
                border-radius: 6px;
            }
            QHeaderView::section {
                background: #f8f9fa;
                padding: 8px;
            }
        """)

        # 可视化画布
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(300)

        # 布局组装
        file_layout.addWidget(self.open_btn, stretch=1)
        file_layout.addWidget(self.load_btn, stretch=1)

        control_layout.addWidget(title)
        control_layout.addWidget(file_group)
        control_layout.addWidget(self.filepathlabel_1)
        control_layout.addWidget(self.info_box)
        control_layout.addStretch()

        vis_splitter.addWidget(self.data_table)
        vis_splitter.addWidget(self.canvas)
        vis_splitter.setSizes([400, 500])

        main_splitter.addWidget(control_panel)
        main_splitter.addWidget(vis_splitter)
        main_splitter.setSizes([300, 700])

        # 主布局设置
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.addWidget(main_splitter)

        # 初始化组件和信号连接
        self._init_widgets()
        self._connect_signals()

    def _init_widgets(self):
        # 初始化Matplotlib画布
        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

    def _connect_signals(self):
        self.open_btn.clicked.connect(self._open_file_dialog)
        self.load_btn.clicked.connect(self._load_dataset)
        self.dataLoaded.connect(self._update_ui)

    def _open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "CSV文件 (*.csv)")
        if path:
            self.filepathlabel_1.setText(f"已选择文件: {os.path.basename(path)}")
            self.current_file = path

    def _load_dataset(self):
        if not hasattr(self, 'current_file'):
            return

        try:
            data = pd.read_csv(self.current_file)
            self.dataLoaded.emit(data)
        except Exception as e:
            self.info_box.append(f"加载文件错误: {str(e)}")

    def _update_ui(self, data):
        # 更新表格数据
        self.model = ELMModel(data)
        self.data_table.setModel(self.model)

        # 显示数据基本信息
        self.info_box.clear()
        self.info_box.append(f"数据维度: {data.shape}")
        self.info_box.append(f"特征列表:\n{', '.join(data.columns)}")

        # 初始化ELM模型
        self.elm_model = elm.ELM(
            input_size=data.shape[1]-1,
            hidden_size=100
        )

    def _plot_results(self, y_true, y_pred):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(y_true, label='实际值')
        ax.plot(y_pred, label='预测值')
        ax.legend()
        self.canvas.draw()

    def _show_metrics(self, rmse):
        self.info_box.append("\n模型性能:")
        self.info_box.append(f"RMSE: {rmse:.4f}")
>>>>>>> 5e7640a (version-1.1)
