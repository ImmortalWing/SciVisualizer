import io
import os
import sys
import traceback
import re

from PySide6.QtCore import QAbstractTableModel, QObject, Qt, Signal, QThread
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCompleter,
    QFileDialog,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,NavigationToolbar2QT
from matplotlib import pyplot as plt
from qfluentwidgets import (
    Action,
    CardWidget,
    CommandBarView,
    ComboBox,
    EditableComboBox,
    Flyout,
    FlyoutAnimationType,
    FluentIcon,
    ImageLabel,
    InfoBar,
    InfoBarPosition,
    Pivot,
    ProgressBar,
    PushButton,
    SegmentedWidget,
    setTheme,
    Theme,
    ToolTipFilter,
    ToolTipPosition,
)

import algorithm.dataanalysisalgo as algo
from tools.pandasmodel import PandasModel


class TextEditStream(QObject):
    text_written = Signal(str)
    
    def __init__(self, textedit):
        super().__init__()
        self.textedit = textedit
        
    def write(self, text):
        self.text_written.emit(str(text))
        
    def flush(self):
        pass

class TrainingWorker(QObject):
    finished = Signal()
    error = Signal(str)
    progress = Signal(str)
    result_ready = Signal(str)
    
    def __init__(self, exp):
        super().__init__()
        self.exp = exp
        self._is_running = True
        
    def run(self):
        try:
            if not self._is_running:
                return
            
            # 创建一个自定义的stdout捕获器来监控训练进度
            original_stdout = sys.stdout
            
            class ProgressCapture:
                def __init__(self, progress_signal):
                    self.progress_signal = progress_signal
                    self.buffer = ""
                
                def write(self, text):
                    # 写入原始stdout以便日志显示
                    original_stdout.write(text)
                    
                    # 累积文本直到换行
                    self.buffer += text
                    if '\n' in text:
                        # 发送完整行作为进度更新
                        lines = self.buffer.split('\n')
                        for line in lines[:-1]:  # 最后一个可能是不完整的行
                            if line.strip():  # 忽略空行
                                self.progress_signal.emit(line)
                        # 保留最后一个不完整的行（如果有）
                        self.buffer = lines[-1] if lines[-1] else ""
                
                def flush(self):
                    original_stdout.flush()
            
            # 替换stdout以捕获进度
            progress_capture = ProgressCapture(self.progress)
            sys.stdout = progress_capture
            
            try:
                # 传递停止检查函数给训练过程
                self.exp.train(stop_check_fn=self.should_stop)
                # 如果训练未被中断，则生成结果图
                if self._is_running:
                    resultpath = self.exp.paint()
                    self.result_ready.emit(resultpath)
                else:
                    print("训练已被用户中断，跳过结果生成阶段")
            finally:
                # 恢复原始stdout
                sys.stdout = original_stdout
                
        except Exception as e:
            self.error.emit(f"Training error: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()
    
    def should_stop(self):
        """检查是否应该停止训练"""
        return not self._is_running
            
    def stop(self):
        """设置停止标志并通知"""
        self._is_running = False
        print("\n正在安全停止训练过程，请稍候...")

class DeepLearningPage(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        
        # 主布局
        self.mainLayout = QVBoxLayout(self)
        self.mainLayout.setContentsMargins(10, 10, 10, 10)
        self.mainLayout.setSpacing(10)
        
        # 页面标题
        self.headerLayout = QHBoxLayout()
        self.pagelabel = QLabel('深度学习')
        self.font = QFont()
        self.font.setPointSize(28)
        self.pagelabel.setFont(self.font)
        self.headerLayout.addWidget(self.pagelabel)
        self.headerLayout.addStretch()
        self.mainLayout.addLayout(self.headerLayout)
        
        # 模型配置卡片
        self.modelConfigCard = CardWidget(self)
        self.modelConfigLayout = QVBoxLayout(self.modelConfigCard)
        
        # 模型选择组
        self.modelSelectionGroup = QGroupBox("模型选择")
        self.modelSelectionLayout = QGridLayout(self.modelSelectionGroup)
        
        # 模型选择下拉框
        self.setmodelcomboBox = ComboBox(self)
        self.setmodelcomboBox.setPlaceholderText("选择一个模型")
        modellist = ['DAR', 'LSTM', 'GRU', 'TCN','Transformer']
        self.setmodelcomboBox.addItems(modellist)
        self.setmodelcomboBox.setCurrentIndex(-1)
        self.setmodelcomboBox.setToolTip("选择要使用的深度学习模型")
        self.setmodelcomboBox.installEventFilter(ToolTipFilter(self.setmodelcomboBox, showDelay=500, position=ToolTipPosition.TOP))
        
        # 模型文件按钮
        self.setmodelbtn5 = PushButton(self)
        self.setmodelbtn5.setText('选择模型文件')
        self.setmodelbtn5.setIcon(FluentIcon.DOCUMENT)
        
        self.modelSelectionLayout.addWidget(QLabel("模型类型:"), 0, 0)
        self.modelSelectionLayout.addWidget(self.setmodelcomboBox, 0, 1)
        self.modelSelectionLayout.addWidget(QLabel("预训练模型:"), 1, 0)
        self.modelSelectionLayout.addWidget(self.setmodelbtn5, 1, 1)
        
        # 参数设置组
        self.paramGroup = QGroupBox("参数设置")
        self.paramLayout = QGridLayout(self.paramGroup)
        
        # 学习率
        self.lrcomboBox = EditableComboBox(self)
        self.lrcomboBox.setPlaceholderText("选择学习率")
        lrlist = ['0.001','0.0005','0.0001']
        self.lrcomboBox.addItems(lrlist)
        self.lrcomboBox.setCurrentIndex(-1)
        self.completer = QCompleter(lrlist, self)
        self.lrcomboBox.setCompleter(self.completer)
        self.lrcomboBox.setToolTip("模型训练的学习率参数")
        self.lrcomboBox.installEventFilter(ToolTipFilter(self.lrcomboBox, showDelay=500))
        
        # 循环次数
        self.epochscomboBox = EditableComboBox(self)
        self.epochscomboBox.setPlaceholderText("选择循环次数")
        epochslist = ['30','50','100']
        self.epochscomboBox.addItems(epochslist)
        self.epochscomboBox.setCurrentIndex(-1)
        self.completer = QCompleter(epochslist, self)
        self.epochscomboBox.setCompleter(self.completer)
        self.epochscomboBox.setToolTip("训练的总循环次数")
        self.epochscomboBox.installEventFilter(ToolTipFilter(self.epochscomboBox, showDelay=500))
        
        # 时间步长
        self.timestepcomboBox = EditableComboBox(self)
        self.timestepcomboBox.setPlaceholderText("选择时间步长")
        timesteplist = ['10','20','30','60']
        self.timestepcomboBox.addItems(timesteplist)
        self.timestepcomboBox.setCurrentIndex(-1)
        self.completer = QCompleter(timesteplist, self)
        self.timestepcomboBox.setCompleter(self.completer)
        self.timestepcomboBox.setToolTip("时序数据的时间步长")
        self.timestepcomboBox.installEventFilter(ToolTipFilter(self.timestepcomboBox, showDelay=500))
        
        # 参数设置按钮
        self.setmodelbtn6 = PushButton(self)
        self.setmodelbtn6.setText('设置默认参数')
        self.setmodelbtn6.setIcon(FluentIcon.SETTING)
        self.setmodelbtn6.clicked.connect(self.setDefaultParams)
        
        self.setmodelbtn8 = PushButton(self)
        self.setmodelbtn8.setText('清除图像')
        self.setmodelbtn8.setIcon(FluentIcon.DELETE)
        self.setmodelbtn8.clicked.connect(self.cleanimage)
        
        self.paramLayout.addWidget(QLabel("学习率:"), 0, 0)
        self.paramLayout.addWidget(self.lrcomboBox, 0, 1)
        self.paramLayout.addWidget(QLabel("循环次数:"), 0, 2)
        self.paramLayout.addWidget(self.epochscomboBox, 0, 3)
        self.paramLayout.addWidget(QLabel("时间步长:"), 1, 0)
        self.paramLayout.addWidget(self.timestepcomboBox, 1, 1)
        self.paramLayout.addWidget(self.setmodelbtn6, 1, 2)
        self.paramLayout.addWidget(self.setmodelbtn8, 1, 3)
        
        # 添加模型选择和参数设置到模型配置卡片
        self.modelConfigLayout.addWidget(self.modelSelectionGroup)
        self.modelConfigLayout.addWidget(self.paramGroup)
        
        # 添加模型配置卡片到主布局
        self.mainLayout.addWidget(self.modelConfigCard)
        
        # 数据操作区域
        self.dataCard = CardWidget(self)
        self.dataLayout = QVBoxLayout(self.dataCard)
        
        # 数据操作按钮组
        self.dataButtonLayout = QHBoxLayout()
        
        self.openfilebtn = PushButton(self)
        self.openfilebtn.setText("打开文件目录")
        self.openfilebtn.setIcon(FluentIcon.FOLDER)
        
        self.readfilebtn = PushButton(self)
        self.readfilebtn.setText("选择数据文件")
        self.readfilebtn.setIcon(FluentIcon.DOCUMENT)
        
        self.filepathlabel_1 = QLabel(self)
        self.filepathlabel_1.setText("未选择文件")
        
        self.trainbtn = PushButton(self)
        self.trainbtn.setText('训练')
        self.trainbtn.setIcon(FluentIcon.PLAY)
        self.trainbtn.clicked.connect(self.usemodeltrain)
        
        self.stopbtn = PushButton(self)
        self.stopbtn.setText('停止训练')
        self.stopbtn.setIcon(FluentIcon.PAUSE)
        self.stopbtn.clicked.connect(self.stop_training)
        self.stopbtn.setEnabled(False)
        
        self.dataButtonLayout.addWidget(self.openfilebtn)
        self.dataButtonLayout.addWidget(self.readfilebtn)
        self.dataButtonLayout.addWidget(self.filepathlabel_1)
        self.dataButtonLayout.addStretch()
        self.dataButtonLayout.addWidget(self.trainbtn)
        self.dataButtonLayout.addWidget(self.stopbtn)
        
        # 进度条
        self.progressBar = ProgressBar(self)
        self.progressBar.setTextVisible(True)
        self.progressBar.setValue(0)
        self.progressBar.setFixedHeight(10)
        
        # 添加按钮组和进度条到数据操作区域
        self.dataLayout.addLayout(self.dataButtonLayout)
        self.dataLayout.addWidget(self.progressBar)
        
        # 添加数据操作区域到主布局
        self.mainLayout.addWidget(self.dataCard)
        
        # 创建分割器用于数据展示区域
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        
        # 数据表格
        self.tableCard = CardWidget(self)
        self.tableLayout = QVBoxLayout(self.tableCard)
        self.tableLabel = QLabel("数据预览")
        self.tableLabel.setAlignment(Qt.AlignCenter)
        self.tableView = QTableView(self)
        self.tableLayout.addWidget(self.tableLabel)
        self.tableLayout.addWidget(self.tableView)
        
        # 日志输出
        self.logCard = CardWidget(self)
        self.logLayout = QVBoxLayout(self.logCard)
        self.logLabel = QLabel("训练日志")
        self.logLabel.setAlignment(Qt.AlignCenter)
        self.textedit = QTextEdit(self)
        self.textedit.setReadOnly(True)
        self.logLayout.addWidget(self.logLabel)
        self.logLayout.addWidget(self.textedit)
        
        # 结果展示
        self.resultCard = CardWidget(self)
        self.resultLayout = QVBoxLayout(self.resultCard)
        self.resultLabel = QLabel("训练结果")
        self.resultLabel.setAlignment(Qt.AlignCenter)
        
        # 创建 Matplotlib 图形和工具栏
        self.figure, self.ax = plt.subplots(figsize=(8, 6), dpi=100)  # 增加图形大小和分辨率
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 允许画布扩展
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # 替换原有的 ImageLabel 为 Matplotlib 图形和工具栏
        self.resultLayout.addWidget(self.resultLabel)
        self.resultLayout.addWidget(self.toolbar)
        self.resultLayout.addWidget(self.canvas, 1)  # 添加拉伸因子使画布占用更多空间
        
        # 添加各个区域到分割器
        self.splitter.addWidget(self.tableCard)
        self.splitter.addWidget(self.logCard)
        self.splitter.addWidget(self.resultCard)
        
        # 设置初始分割比例 - 给结果区域更多空间
        self.splitter.setSizes([250, 250, 500])
        
        # 添加分割器到主布局
        self.mainLayout.addWidget(self.splitter, 1)  # 1表示拉伸因子
        
        # 连接信号和槽
        self.parentconnection()
        
    def parentconnection(self):
        self.readfilebtn.clicked.connect(self.readfile)
        self.openfilebtn.clicked.connect(self.openfilepath)
        self.setmodelcomboBox.currentTextChanged.connect(self.onModelChanged)

    def onModelChanged(self, model_name):
        """当模型选择改变时更新UI"""
        if model_name:
            self.show_info_bar('模型已选择', f'当前选择的模型: {model_name}', 'success')
            # 自动为该模型设置推荐参数
            self.setDefaultParams()

    def setDefaultParams(self):
        """设置默认参数"""
        model = self.setmodelcomboBox.currentText()
        
        # 根据不同模型设置不同的默认参数
        if model == 'LSTM' or model == 'GRU':
            self.lrcomboBox.setCurrentText('0.001')
            self.epochscomboBox.setCurrentText('50')
            self.timestepcomboBox.setCurrentText('20')
        elif model == 'Transformer':
            self.lrcomboBox.setCurrentText('0.0001')
            self.epochscomboBox.setCurrentText('100')
            self.timestepcomboBox.setCurrentText('30')
        else:  # 默认参数
            self.lrcomboBox.setCurrentText('0.001')
            self.epochscomboBox.setCurrentText('30')
            self.timestepcomboBox.setCurrentText('10')
            
        self.show_info_bar('参数已设置', f'已为{model}模型设置默认参数', 'success')

    def openfilepath(self):
        start_directory = 'data'
        os.system("explorer.exe %s" % start_directory)

    def readfile(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open file', './data')
        if not self.fname[0]:  # 用户取消了选择
            return
        
        self.filepathlabel_1.setText("文件路径：" + self.fname[0])
        try:
            self.df = algo.readfile(self.fname[0])
            pandasmodel = PandasModel(self.df)
            self.tableView.setModel(pandasmodel)
            
            # 显示基本数据统计信息
            stats = self.df.describe()
            print("数据基本统计：")
            print(stats)
            
            self.show_info_bar('文件已加载', '成功加载数据文件', 'success')
        except Exception as e:
            self.show_info_bar('加载失败', f'无法加载文件: {str(e)}', 'error', 4000)

    def usemodeltrain(self):
        # 参数验证
        if not self.validate_training_params():
            return
        
        from algorithm.deeplearning.exp import Exp
        
        # 清空日志
        self.textedit.clear()
        
        # 创建并连接流
        self.stream = TextEditStream(self.textedit)
        self.stream.text_written.connect(self.textedit.append)
        
        # 重定向stdout
        self.old_stdout = sys.stdout
        sys.stdout = self.stream

        # 创建所需目录
        self._ensure_directories_exist()

        # 配置实验
        self.exp = Exp()
        print(f"选择的模型: {self.setmodelcomboBox.currentText()}")
        self.exp.setmodel(self.setmodelcomboBox.currentText())
        self.exp.config.learning_rate = float(self.lrcomboBox.currentText())
        self.exp.config.epochs = int(self.epochscomboBox.currentText())
        self.exp.config.timestep = int(self.timestepcomboBox.currentText())

        # 更新UI状态
        self._update_ui_for_training_start()
        
        # 创建worker和线程
        self._setup_training_thread()
        
        # 显示训练开始通知
        self.show_info_bar('训练开始', f'开始训练{self.setmodelcomboBox.currentText()}模型', 'success')

    def _ensure_directories_exist(self):
        """确保所需的目录存在"""
        directory_paths = ["algorithm/deeplearning/result/pic",
                          "algorithm/deeplearning/result/model",
                          "algorithm/deeplearning/result/pre&loss",
                          'algorithm/deeplearning/result/loss']
        for path in directory_paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def _update_ui_for_training_start(self):
        """更新UI以反映训练开始状态"""
        self.trainbtn.setEnabled(False)
        self.trainbtn.setText('训练中...')
        self.stopbtn.setEnabled(True)
        
        # 设置进度条
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.progressBar.setFormat("准备训练...")

    def _setup_training_thread(self):
        """设置训练线程和工作器"""
        # 创建worker和线程
        self.worker = TrainingWorker(self.exp)
        self.thread = QThread()
        
        # 将worker移动到线程
        self.worker.moveToThread(self.thread)
        
        # 连接信号
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self._handle_training_error)
        self.worker.result_ready.connect(self._handle_training_result)
        self.worker.progress.connect(self._update_progress)
        
        # 启动线程
        self.thread.start()

    def _update_progress(self, progress_text):
        """更新训练进度"""
        try:
            # 尝试从进度文本中提取进度百分比
            if "epoch" in progress_text.lower():
                # 查找格式为 "Epoch [X/Y]" 的文本
                match = re.search(r"Epoch \[(\d+)/(\d+)\]", progress_text)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    percent = int(current * 100 / total)
                    self.progressBar.setValue(percent)
                    
                    # 更新进度条文本
                    self.progressBar.setFormat(f"训练进度: {percent}% (Epoch {current}/{total})")
                    
                    # 在日志中显示当前进度
                    print(f"进度更新: {percent}%")
        except Exception as e:
            print(f"进度解析错误: {str(e)}")
            # 如果无法解析进度，则忽略

    def _handle_training_result(self, resultpath):
        """处理训练结果并更新图形"""
        # 清空当前图形
        self.ax.clear()
        
        # 加载并显示结果图像
        img = plt.imread(resultpath)
        self.ax.imshow(img)
        self.ax.axis('off')  # 隐藏坐标轴
        self.figure.tight_layout()  # 优化图形布局，减少边距
        
        # 更新画布
        self.canvas.draw()
        
        # 清理训练状态
        self._cleanup_training()
        
        # 显示训练完成通知
        self.show_info_bar('训练完成', '模型训练已成功完成', 'success')

    def _handle_training_error(self, error):
        """处理训练过程中的错误"""
        print(error)
        
        # 清理训练状态
        self._cleanup_training()
        
        # 尝试显示部分训练结果
        self._display_partial_results()
        
        # 显示错误通知
        self.show_info_bar('训练错误', '训练过程中发生错误，已显示部分结果', 'error', 4000)

    def _cleanup_training(self, stopped=False):
        """清理训练相关资源
        
        参数:
            stopped (bool): 是否是由用户手动停止的训练
        """
        # 恢复stdout并重置UI状态
        if hasattr(self, 'old_stdout'):
            sys.stdout = self.old_stdout
            
        self.trainbtn.setEnabled(True)
        self.trainbtn.setText('训练')
        self.stopbtn.setEnabled(False)
        self.stopbtn.setText('停止训练')
        
        # 设置进度条状态
        if stopped:
            self.progressBar.setFormat("训练已停止")
        else:
            self.progressBar.setValue(100)
            self.progressBar.setFormat("训练完成")
        
        # 清理线程和worker
        if hasattr(self, 'thread'):
            # 确保线程已经退出
            if self.thread.isRunning():
                self.thread.quit()
                self.thread.wait(2000)  # 等待最多2秒
            del self.thread
            
        if hasattr(self, 'worker'):
            del self.worker
            
        # 显示停止通知
        if stopped:
            print("训练已被用户终止")

    def stop_training(self):
        """停止正在进行的训练任务"""
        if not hasattr(self, 'worker') or not hasattr(self, 'thread'):
            return
            
        if not self.thread.isRunning():
            self.show_info_bar('无需操作', '当前没有正在进行的训练任务', 'info')
            return
            
        # 更新UI
        self.stopbtn.setEnabled(False)
        self.stopbtn.setText('正在停止...')
        self.progressBar.setFormat("正在安全停止训练...")
        
        # 通知用户
        self.show_info_bar('停止中', '正在安全停止训练，请稍候...', 'warning', 3000)
        
        # 设置停止标志
        self.worker.stop()
        
        # 等待线程自然结束
        QApplication.processEvents()  # 确保UI更新
        
        # 设置超时机制以防止无限等待
        stop_timeout = 5000  # 5秒超时
        if not self.thread.wait(stop_timeout):
            print("警告: 训练线程未在预期时间内停止")
            self.show_info_bar('警告', '训练线程未能及时响应停止命令', 'warning')
            # 在这里我们不使用terminate()，而是继续等待线程自然结束
        
        # 清理资源
        self._cleanup_training(stopped=True)
        
        # 尝试显示部分训练结果（如果有）
        self._display_partial_results()

    def _display_partial_results(self):
        """显示部分训练结果，即使在训练被中断的情况下"""
        if hasattr(self, 'exp') and hasattr(self.exp, 'loss_save1') and len(self.exp.loss_save1) > 0:
            # 清空当前图形
            self.ax.clear()
            
            # 创建部分训练结果的可视化
            epochs = range(1, len(self.exp.loss_save1) + 1)
            
            # 绘制训练和验证损失曲线
            self.ax.plot(epochs, self.exp.loss_save1, 'b-', label='训练损失')
            if hasattr(self.exp, 'loss_save2') and len(self.exp.loss_save2) > 0:
                self.ax.plot(epochs, self.exp.loss_save2, 'r-', label='验证损失')
                
            # 添加标题和标签
            self.ax.set_title(f'{self.setmodelcomboBox.currentText()}模型训练过程 (中断于第{len(epochs)}个轮次)', fontproperties='SimHei')
            self.ax.set_xlabel('轮次 (Epoch)')
            self.ax.set_ylabel('损失 (Loss)')
            self.ax.legend()
            self.ax.grid(True)
            
            # 更新画布
            self.figure.tight_layout()
            self.canvas.draw()
            
            print(f"已显示中断前的{len(epochs)}个训练轮次的损失曲线")

    def cleanimage(self):
        # 修正方法，使用 Matplotlib 图形而不是旧的 GraphicsView
        if hasattr(self, 'ax'):
            self.ax.clear()
            self.ax.set_axis_off()
            self.canvas.draw()
            self.show_info_bar('图像已清除', '结果图像已被清除', 'info')

    def showCommandBar(self):
        view = CommandBarView(self)

        view.addAction(Action(FluentIcon.SHARE, '分享'))
        view.addAction(Action(FluentIcon.SAVE, '保存'))
        view.addAction(Action(FluentIcon.DELETE, '删除'))

        view.addHiddenAction(Action(FluentIcon.APPLICATION, '应用', shortcut='Ctrl+A'))
        view.addHiddenAction(Action(FluentIcon.SETTING, '设置', shortcut='Ctrl+S'))
        view.resizeToSuitableWidth()

        Flyout.make(view, self.graphicsView, self, FlyoutAnimationType.FADE_IN)

    def validate_training_params(self):
        """验证训练参数是否完整有效
        
        返回:
            bool: 参数是否有效
        """
        # 检查模型选择
        if not self.setmodelcomboBox.currentText():
            self.show_info_bar('参数错误', '请选择一个模型', 'warning', 3000)
            return False
        
        # 检查训练参数
        if not self.lrcomboBox.currentText() or not self.epochscomboBox.currentText() or not self.timestepcomboBox.currentText():
            self.show_info_bar('参数错误', '请设置所有训练参数', 'warning', 3000)
            return False
        
        # 检查数据
        if not hasattr(self, 'df'):
            self.show_info_bar('数据错误', '请先加载数据文件', 'warning', 3000)
            return False
        
        # 验证参数值的合法性
        try:
            lr = float(self.lrcomboBox.currentText())
            epochs = int(self.epochscomboBox.currentText())
            timestep = int(self.timestepcomboBox.currentText())
            
            if lr <= 0 or epochs <= 0 or timestep <= 0:
                self.show_info_bar('参数错误', '学习率、轮次和时间步长必须为正数', 'warning', 3000)
                return False
        except ValueError:
            self.show_info_bar('参数错误', '无效的参数值格式', 'warning', 3000)
            return False
        
        return True

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
