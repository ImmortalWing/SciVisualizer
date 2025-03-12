import os
from PySide6.QtCore import Qt, QAbstractTableModel, Signal
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, \
    QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene, QHeaderView, QSplitter
from PySide6.QtGui import QFont
from qfluentwidgets import PushButton, FluentIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import pandas as pd
import matplotlib.pyplot as plt
from algorithm import elm_algo as elm
import numpy as np

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
        self.demo_btn = PushButton(FluentIcon.PLAY, "运行示例", self)  # 新增按钮定义
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
        toolbar = NavigationToolbar2QT(self.canvas, self)  # 添加工具栏

        # 布局组装
        file_layout.addWidget(self.open_btn, stretch=1)
        file_layout.addWidget(self.load_btn, stretch=1)
        file_layout.addWidget(self.demo_btn, stretch=1)

        control_layout.addWidget(title)
        control_layout.addWidget(file_group)
        control_layout.addWidget(self.filepathlabel_1)
        control_layout.addWidget(self.info_box)
        control_layout.addStretch()

        vis_splitter.addWidget(self.data_table)
        vis_splitter.addWidget(toolbar)  # 添加工具栏到布局
        vis_splitter.addWidget(self.canvas)
        vis_splitter.setSizes([400, 30, 500])

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
        # 初始化Matplotlib画布 - 这里不需要重复初始化，因为在_init_ui中已经初始化过了
        pass

    def _connect_signals(self):
        self.open_btn.clicked.connect(self._open_file_dialog)
        self.load_btn.clicked.connect(self._load_dataset)
        self.demo_btn.clicked.connect(self._run_demo)
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
            input_size=data.shape[1] - 1,
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

    def _run_demo(self):
        """执行ELM示例演示"""
        from PySide6.QtCore import QThread, Signal
        
        class DemoThread(QThread):
            update_progress = Signal(str)
            show_result = Signal(float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            error_occurred = Signal(str)

            def run(self):
                try:
                    self.update_progress.emit("正在生成示例数据...")
                    # 使用更复杂的函数生成数据，降低噪声水平
                    X, y = elm.ELM(1, 1).generate_example_data(
                        n_samples=2000, 
                        noise_level=0.05,  # 降低噪声水平
                        function='complex'  # 使用更复杂的函数
                    )
                    
                    # 数据标准化
                    X_mean, X_std = X.mean(), X.std()
                    y_mean, y_std = y.mean(), y.std()
                    X_norm = (X - X_mean) / X_std
                    y_norm = (y - y_mean) / y_std
                    
                    # 分割数据集 - 增加训练集比例
                    split_idx = int(0.8 * len(X))  # 80%用于训练
                    X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
                    y_train, y_test = y_norm[:split_idx], y_norm[split_idx:]

                    self.update_progress.emit("正在初始化ELM模型...")
                    # 使用优化后的ELM模型
                    model = elm.ELM(
                        input_size=1,
                        hidden_size=300,  # 增加隐藏层神经元数量
                        activation='tanh',  # 使用tanh激活函数
                        C=0.01  # 添加轻微正则化
                    )

                    self.update_progress.emit("正在训练模型...")
                    # 训练模型
                    model.fit(X_train, y_train)

                    self.update_progress.emit("正在进行预测...")
                    # 预测并计算性能
                    y_pred = model.predict(X_test)
                    rmse = model.score(X_test, y_test)
                    r2 = model.r2_score(X_test, y_test)
                    
                    self.update_progress.emit(f"模型训练完成! RMSE: {rmse:.4f}, R²: {r2:.4f}")
                    
                    # 还原预测值和真实值到原始尺度
                    y_test_orig = y_test * y_std + y_mean
                    y_pred_orig = y_pred * y_std + y_mean
                    
                    # 创建示例数据表 - 按输入特征排序以便更好地可视化
                    indices = np.argsort(X_test.flatten())
                    X_test_sorted = X_test.flatten()[indices]
                    y_test_sorted = y_test[indices]
                    y_pred_sorted = y_pred[indices]
                    
                    # 转换回原始尺度用于显示
                    X_orig_sorted = X_test_sorted * X_std + X_mean
                    y_test_orig_sorted = y_test_sorted * y_std + y_mean
                    y_pred_orig_sorted = y_pred_sorted * y_std + y_mean
                    
                    demo_data = pd.DataFrame({
                        'X': X_orig_sorted,
                        'Y真实值': y_test_orig_sorted,
                        'Y预测值': y_pred_orig_sorted,
                        '误差': np.abs(y_test_orig_sorted - y_pred_orig_sorted)
                    })

                    self.show_result.emit(rmse, r2, y_test_sorted, y_pred_sorted, X_test_sorted, demo_data)
                    
                except Exception as e:
                    import traceback
                    self.error_occurred.emit(f"示例运行失败: {str(e)}\n{traceback.format_exc()}")

        # 创建并启动线程
        self.info_box.clear()
        self.info_box.append("🚀 开始运行ELM示例演示...")
        self.thread = DemoThread()
        self.thread.update_progress.connect(lambda msg: self.info_box.append(msg))
        self.thread.show_result.connect(self._handle_demo_result)
        self.thread.error_occurred.connect(self._show_error_dialog)
        self.thread.start()

    def _handle_demo_result(self, rmse, r2, y_true, y_pred, x_test, demo_data):
        """处理演示结果"""
        # 显示性能指标
        self.info_box.append(f"\n✅ 模型训练完成！")
        self.info_box.append(f"📊 测试集RMSE: {rmse:.4f}")
        self.info_box.append(f"📈 决定系数R²: {r2:.4f}")
        self.info_box.append(f"📉 最大误差: {np.max(np.abs(y_true - y_pred)):.4f}")
        self.info_box.append(f"📉 平均误差: {np.mean(np.abs(y_true - y_pred)):.4f}")
        
        # 更新表格显示
        self.model = ELMModel(demo_data)
        self.data_table.setModel(self.model)
        
        # 更新图表
        self._update_plot(x_test, y_true, y_pred, r2)
        
        # 显示结果摘要
        from qfluentwidgets import MessageBox
        MessageBox.success(
            self,
            "演示完成",
            f"成功完成ELM示例演示\n测试集RMSE: {rmse:.4f}\nR²: {r2:.4f}",
            parent=self
        )

    def _update_plot(self, x_test, y_true, y_pred, r2):
        """更新预测结果可视化"""
        self.figure.clf()
        
        # 创建两个子图
        ax1 = self.figure.add_subplot(211)  # 上方子图：预测vs真实
        ax2 = self.figure.add_subplot(212)  # 下方子图：误差分布
        
        # 上方子图：预测vs真实值 - 改进可视化效果
        ax1.scatter(x_test, y_true, color='blue', alpha=0.5, label='实际值', s=30)
        ax1.plot(x_test, y_pred, 'r-', linewidth=2, label='预测值')
        
        ax1.set_title(f"ELM模型预测效果 (R² = {r2:.4f})", fontsize=12, fontfamily='Microsoft YaHei')
        ax1.set_xlabel("输入特征", fontfamily='Microsoft YaHei')
        ax1.set_ylabel("标准化输出", fontfamily='Microsoft YaHei')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 下方子图：误差分布
        errors = y_pred - y_true
        ax2.hist(errors, bins=30, alpha=0.7, color='green')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_title(f"预测误差分布 (均值={errors.mean():.4f}, 标准差={errors.std():.4f})", 
                     fontsize=12, fontfamily='Microsoft YaHei')
        ax2.set_xlabel("预测误差", fontfamily='Microsoft YaHei')
        ax2.set_ylabel("频数", fontfamily='Microsoft YaHei')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # 调整布局
        self.figure.tight_layout()
        self.canvas.draw()

    def _show_error_dialog(self, msg):
        """显示错误对话框"""
        from qfluentwidgets import MessageBox
        MessageBox.critical(
            self,
            "运行错误",
            msg,
            parent=self
        )
        
if __name__ == "__main__":
    app = QApplication([])
    page = ELMPage("ELM Page")
    page.show()
    app.exec()