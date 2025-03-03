import os
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, \
    QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene

from qfluentwidgets import Pivot, setTheme, Theme, SegmentedWidget, FluentIcon, PushButton
import algorithm.dataanalysisalgo as algo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class DataAnalysisPage(QWidget):
<<<<<<< HEAD
    def __init__(self):
        super().__init__()
=======
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
>>>>>>> 5e7640a (version-1.1)
        self.setStyleSheet("""
            Demo{background: white}
            QLabel{
                font: 20px 'Segoe UI';
                background: rgb(242,242,242);
                border-radius: 8px;
            }
        """)
        # self.resize(400, 400)

        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)
        self.vBoxLayout = QVBoxLayout(self)

        self.dsInterface = Page1('D-S', self)
        self.pcaInterface = Page2('Album Interface', self)
        self.factorInterface = Page3('Artist Interface', self)
        self.nullInterface = Page4('nullInterface', self)

        # add items to pivot
        self.addSubInterface(self.dsInterface, 'D-SInterface', 'D-S证据理论')
        self.addSubInterface(self.pcaInterface, 'pcaInterface', '主成分分析')
        self.addSubInterface(self.factorInterface, 'factorInterface', '因子分析')
        self.addSubInterface(self.nullInterface, 'nullInterface', '待定')

        self.vBoxLayout.addWidget(self.pivot)
        self.vBoxLayout.addWidget(self.stackedWidget)
        self.vBoxLayout.setContentsMargins(30, 10, 30, 30)

        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.dsInterface)
        self.pivot.setCurrentItem(self.dsInterface.objectName())

    def addSubInterface(self, widget: QWidget, objectName, text):
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget),
        )

    def onCurrentIndexChanged(self, index):
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())


class Page0(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.df = 0
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout1.setObjectName(u"horizontalLayout1")
        self.verticalLayout.addLayout(self.horizontalLayout1)
        self.textedit = QTextEdit(self)
        # self.textedit.setMinimumSize(0, 500)
        self.tableView = QTableView(self)
        self.graphicsView = QGraphicsView(self)

        self.horizontalLayout1.addWidget(self.textedit)
        self.horizontalLayout1.addWidget(self.graphicsView)
        self.horizontalLayout1.addWidget(self.tableView)

        self.horizontalLayout2 = QHBoxLayout()
        self.horizontalLayout2.setObjectName(u"horizontalLayout2")
        self.verticalLayout.addLayout(self.horizontalLayout2)

        self.openfilebtn = PushButton(self)
        self.openfilebtn.setText("打开文件目录")
        self.horizontalLayout2.addWidget(self.openfilebtn)

        self.readfilebtn = PushButton(self)
        self.readfilebtn.setText("读取数据文件")
        self.horizontalLayout2.addWidget(self.readfilebtn)
        self.filepathlabel_1 = QLabel(self)
        self.horizontalLayout2.addWidget(self.filepathlabel_1)  # 添加到布局中

        self.parentconnection()

    def parentconnection(self):
        self.readfilebtn.clicked.connect(self.readfile)
        self.openfilebtn.clicked.connect(self.openfilepath)

    def readfile(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open file', './data')
<<<<<<< HEAD
        self.filepathlabel_1.setText("文件路径：" + self.fname[0])
=======
        self.filepathlabel_1.setText(f"文件路径：{self.fname[0]}" )
>>>>>>> 5e7640a (version-1.1)
        self.df = algo.readfile(self.fname[0])
        pandasmodel = pandasModel(self.df)
        self.tableView.setModel(pandasmodel)

    def openfilepath(self):
        start_directory = 'data'
        os.system("explorer.exe %s" % start_directory)


class Page1(Page0):
    def __init__(self, text: str, parent=None):
        super().__init__(text)
        self.connection()

    def calds(self):
        ds = str(algo.calds(self.df))
        # 绘图 柱状图
        self.textedit.setText(ds)

    def connection(self):
        self.readfilebtn.clicked.connect(lambda: self.calds())


class Page2(Page0):
    def __init__(self, text: str, parent=None):
        super().__init__(text)
        self.connection()

    def calcpca(self):
        outpca = algo.calcpca(self.df)
        # self.textEdit.setText(str(outpca))
        # 绘图
        # 实例化MplCanvas
        self.sc = MplCanvas(self, width=6, height=5)
        x = outpca
        self.sc.axes.plot(x)
        self.sc.axes.set_title('pca')
        self.graphic_scene = QGraphicsScene()
        self.graphic_scene.addWidget(self.sc)
        self.graphicsView.setScene(self.graphic_scene)
        self.graphicsView.show()
        self.textedit.setText(str(outpca))

    def connection(self):
        self.readfilebtn.clicked.connect(lambda: self.calcpca())


class Page3(Page0):
    def __init__(self, text: str, parent=None):
        super().__init__(text)
        self.connection()

    def calfactor(self):
        self.sc = MplCanvas(self, width=6, height=5)
        self.ev, self.v = algo.calfactor(self.df)
        self.sc.axes.scatter(range(1, self.df.shape[1] + 1), self.ev)
        self.sc.axes.plot(range(1, self.df.shape[1] + 1), self.ev)
        self.sc.axes.set_title("Scree Plot")
        # self.sc.axes.xlabel("Factors")
        # self.sc.axes.ylabel("Eigenvalue")
        self.graphic_scene = QGraphicsScene()
        self.graphic_scene.addWidget(self.sc)
        self.graphicsView.setScene(self.graphic_scene)
        self.graphicsView.show()
        self.textedit.setText(str(self.ev) + str(self.v))

    def connection(self):
        self.readfilebtn.clicked.connect(lambda: self.calfactor())


class Page4(Page0):
    def __init__(self, text: str, parent=None):
        super().__init__(text)


class pandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)  # 添加子图
        super(MplCanvas, self).__init__(fig)
