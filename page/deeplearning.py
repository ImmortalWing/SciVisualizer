import os,sys
from PySide6.QtCore import Qt, QAbstractTableModel,Slot,QProcess
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, \
    QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene,QGridLayout,QCompleter
from PySide6.QtGui import QFont,QPixmap
from qfluentwidgets import Pivot, setTheme, Theme, SegmentedWidget, FluentIcon, PushButton,ComboBox,EditableComboBox
import algorithm.dataanalysisalgo as algo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from tools.pandasmodel import PandasModel
from algorithm.deeplearning import exp


# 支持向量机
class DeepLearningPage(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        #初始化布局
        self.setObjectName(text.replace(' ', '-'))
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.confighLayout = QGridLayout()
        self.confighLayout.setObjectName(u"confighLayout")
        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout1.setObjectName(u"horizontalLayout1")
        self.horizontalLayout2 = QHBoxLayout()
        self.horizontalLayout2.setObjectName(u"horizontalLayout2")
        self.terminal_edit = QTextEdit()
        self.terminal_edit.setReadOnly(True)
        self.terminal_edit.setMinimumSize(0, 400)
        self.horizontalLayout3 = QHBoxLayout()
        self.horizontalLayout3.setObjectName(u"horizontalLayout3")

        self.pagelabel = QLabel('深度学习')
        self.font = QFont()
        self.font.setPointSize(28)
        self.pagelabel.setFont(self.font)
        self.verticalLayout.addWidget(self.pagelabel)
        self.verticalLayout.addLayout(self.confighLayout)
        self.verticalLayout.addLayout(self.horizontalLayout1)
        self.verticalLayout.addLayout(self.horizontalLayout2)
        self.verticalLayout.addWidget(self.terminal_edit)
        self.verticalLayout.addLayout(self.horizontalLayout3)

        #添加控件
        self.setmodelcomboBox = ComboBox(self)
        self.setmodelcomboBox.setPlaceholderText("选择一个模型")
        modellist = ['DAR', 'LSTM', 'GRU', 'TCN','Transformer','CNN']
        self.setmodelcomboBox.addItems(modellist)
        self.setmodelcomboBox.setCurrentIndex(-1)
        #self.setmodelcomboBox.currentTextChanged.connect(print)


        self.lrcomboBox = EditableComboBox(self)
        self.lrcomboBox.setPlaceholderText("选择学习率")
        lrlist = ['0.001','0.0005','0.0001']
        self.lrcomboBox.addItems(lrlist)
        self.lrcomboBox.setCurrentIndex(-1)
        self.completer = QCompleter(lrlist, self)
        self.lrcomboBox.setCompleter(self.completer)

        
        self.epochscomboBox = EditableComboBox(self)
        self.epochscomboBox.setPlaceholderText("选择循环次数")
        epochslist = ['30','50','100']
        self.epochscomboBox.addItems(epochslist)
        self.epochscomboBox.setCurrentIndex(-1)
        self.completer = QCompleter(epochslist, self)
        self.epochscomboBox.setCompleter(self.completer)


        self.timestepcomboBox = EditableComboBox(self)
        self.timestepcomboBox.setPlaceholderText("选择时间步长")
        timesteplist = ['10','20','30','60']
        self.timestepcomboBox.addItems(timesteplist)
        self.timestepcomboBox.setCurrentIndex(-1)
        self.completer = QCompleter(timesteplist, self)
        self.timestepcomboBox.setCompleter(self.completer)


        self.initmodelbtn = PushButton(self)
        self.initmodelbtn.setText('初始化模型')

        self.trainmodelbtn = PushButton(self)
        self.trainmodelbtn.setText('训练模型')

        self.usemodelbtn = PushButton(self)
        self.usemodelbtn.setText('预测')

        self.setmodelbtn8 = PushButton(self)
        self.confighLayout.addWidget(self.setmodelcomboBox,0,0)
        self.confighLayout.addWidget(self.lrcomboBox,0,1)
        self.confighLayout.addWidget(self.epochscomboBox,0,2)
        self.confighLayout.addWidget(self.timestepcomboBox,0,3)
        self.confighLayout.addWidget(self.initmodelbtn,1,0)
        self.confighLayout.addWidget(self.trainmodelbtn,1,1)
        self.confighLayout.addWidget(self.usemodelbtn,1,2)
        self.confighLayout.addWidget(self.setmodelbtn8,1,3)

        self.textedit = QTextEdit(self)
        # self.textedit.setMinimumSize(0, 500)
        self.tableView = QTableView(self)
        self.scene = QGraphicsScene(self)
        self.graphicsView = QGraphicsView(self.scene)

        self.openfilebtn = PushButton(self)
        self.openfilebtn.setText("打开文件目录")
        self.horizontalLayout1.addWidget(self.openfilebtn)

        self.readfilebtn = PushButton(self)
        self.readfilebtn.setText("选择数据文件")
        self.horizontalLayout1.addWidget(self.readfilebtn)
        self.filepathlabel_1 = QLabel(self)
        self.horizontalLayout1.addWidget(self.filepathlabel_1)  # 添加到布局中
        
        self.horizontalLayout2.addWidget(self.tableView)
        self.horizontalLayout2.addWidget(self.textedit)
        self.horizontalLayout2.addWidget(self.graphicsView)

        self.savepredatabtn = PushButton(self)
        self.savepredatabtn.setText('保存预测结果')
        self.saveimgbtn = PushButton(self)
        self.saveimgbtn.setText('保存图像')
        self.cleartextbtn = PushButton(self)
        self.cleartextbtn.setText('清屏')
        self.horizontalLayout3.addWidget(self.savepredatabtn)
        self.horizontalLayout3.addWidget(self.saveimgbtn)
        self.horizontalLayout3.addWidget(self.cleartextbtn)

        # 初始化终端控件
        # 保存原始的标准输出对象
        self.original_stdout = sys.stdout
        # 将标准输出重定向到自定义的写入函数
        sys.stdout = self

        self.parentconnection()
    def parentconnection(self):
        self.readfilebtn.clicked.connect(self.readfile)
        self.openfilebtn.clicked.connect(self.openfilepath)
        self.initmodelbtn.clicked.connect(self.initexp)
        self.trainmodelbtn.clicked.connect(self.trainexp)
        self.cleartextbtn.clicked.connect(self.cleartext)
        

    def openfilepath(self):
        start_directory = 'data'
        os.system("explorer.exe %s" % start_directory)

    def readfile(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open file', './data')
        self.filepathlabel_1.setText("文件路径：" + self.fname[0])
        self.df = algo.readfile(self.fname[0])
        pandasmodel = PandasModel(self.df)
        self.tableView.setModel(pandasmodel)

    def write(self, text):
        """
        这个函数会在print等操作向标准输出写内容时被调用
        """
        self.terminal_edit.append(text)
        # 可以选择是否还将内容输出到原始的标准输出，比如控制台
        # self.original_stdout.write(text)

    def flush(self):
        """
        刷新操作，确保输出及时显示
        """
        pass

    def initexp(self):
        self.exp1 = exp.Exp()
        self.exp1.model = self.setmodelcomboBox.currentText()
        self.exp1.learning_rate = float(self.lrcomboBox.currentText())
        self.exp1.epoch = int(self.epochscomboBox.currentText())
        self.exp1.timestep = int(self.timestepcomboBox.currentText())

        def check_directory(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"文件夹 '{directory}' 不存在，已成功创建。")
            else:
                print(f"文件夹 '{directory}' 已存在。")

        # 指定要检查的目录路径
        directory_paths = ["algorithm/deeplearning/result/pic", "algorithm/deeplearning/result/model", "algorithm/deeplearning/result/pre&loss", 'algorithm/deeplearning/result/loss']
        for path in directory_paths:
            # 调用函数检查并创建目录
            check_directory(path)
        print('初始化模型完毕')
    def trainexp(self):
        self.exp1.setmodel(self.exp1.model)
        self.exp1.train()
        graphic_path = self.exp1.paint()
        pixmap = QPixmap(graphic_path)
        self.scene.addPixmap(pixmap)

    def cleartext(self):
        self.terminal_edit.setText('')