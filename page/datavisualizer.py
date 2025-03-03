import os
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, \
<<<<<<< HEAD
    QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene

from qfluentwidgets import Pivot, setTheme, Theme, SegmentedWidget, FluentIcon, PushButton
import algorithm.dataanalysisalgo as algo
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class DataVisualizerPage(QWidget):
    def __init__(self):
        super().__init__()
        self.vBoxLayout = QVBoxLayout(self)

=======
    QTextEdit, QTableView, QGraphicsView, QFileDialog, QGraphicsScene, QMessageBox, QComboBox

from qfluentwidgets import Pivot, setTheme, Theme, SegmentedWidget, FluentIcon, PushButton,ComboBox,InfoBar,InfoBarPosition
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from config.database import execute_query
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.font_manager as fm

# 添加字体设置
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
# 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class DataVisualizer(QWidget):
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(text.replace(' ', '-'))
        self.setStyleSheet("""
            Demo{background: white}
            QLabel{
                font: 20px 'Segoe UI';
                background: rgb(242,242,242);
                border-radius: 8px;
            }
        """)

        # 初始化界面布局
        self.pivot = SegmentedWidget(self)
        self.stackedWidget = QStackedWidget(self)
        self.vBoxLayout = QVBoxLayout(self)

        # 创建各个功能页面
        self.dataTableInterface = DataTablePage('数据表格', self)
        self.chartInterface = ChartPage('图表可视化', self)
        self.statInterface = StatisticsPage('统计分析', self)

        # 添加页面到切换组件
        self.addSubInterface(self.dataTableInterface, 'DataTable-Interface', '数据表格')
        self.addSubInterface(self.chartInterface, 'Chart-Interface', '图表可视化')
        self.addSubInterface(self.statInterface, 'Stat-Interface', '统计分析')

        # 设置布局
        self.vBoxLayout.addWidget(self.pivot)
        self.vBoxLayout.addWidget(self.stackedWidget)
        self.vBoxLayout.setContentsMargins(30, 10, 30, 30)

        # 连接信号和槽
        self.stackedWidget.currentChanged.connect(self.onCurrentIndexChanged)
        self.stackedWidget.setCurrentWidget(self.dataTableInterface)
        self.pivot.setCurrentItem(self.dataTableInterface.objectName())

    def addSubInterface(self, widget: QWidget, objectName, text):
        """添加子界面到切换组件"""
        widget.setObjectName(objectName)
        self.stackedWidget.addWidget(widget)
        self.pivot.addItem(
            routeKey=objectName,
            text=text,
            onClick=lambda: self.stackedWidget.setCurrentWidget(widget),
        )

    def onCurrentIndexChanged(self, index):
        """切换页面时更新pivot选中状态"""
        widget = self.stackedWidget.widget(index)
        self.pivot.setCurrentItem(widget.objectName())


class BasePage(QWidget):
    """所有页面的基类，包含共享功能"""
    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.df = None
        self.canvas = None
        self.graphic_scene = None
        
        # 创建基本布局
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setObjectName(u"verticalLayout")
        
        # 上半部分布局: 结果显示区
        self.horizontalLayout1 = QHBoxLayout()
        self.horizontalLayout1.setObjectName(u"horizontalLayout1")
        self.verticalLayout.addLayout(self.horizontalLayout1)
        
        # 文本显示区
        self.textedit = QTextEdit(self)
        self.textedit.setReadOnly(True)
        
        # 图形显示区
        self.graphicsView = QGraphicsView(self)
        
        # 表格显示区
        self.tableView = QTableView(self)
        
        # 添加组件到布局
        self.horizontalLayout1.addWidget(self.textedit)
        self.horizontalLayout1.addWidget(self.graphicsView)
        self.horizontalLayout1.addWidget(self.tableView)
        
        # 下半部分布局: 操作按钮区
        self.horizontalLayout2 = QHBoxLayout()
        self.horizontalLayout2.setObjectName(u"horizontalLayout2")
        self.verticalLayout.addLayout(self.horizontalLayout2)
        
        # 数据库选择按钮
        self.db_select_btn = PushButton(self)
        self.db_select_btn.setText("选择数据库")
        self.horizontalLayout2.addWidget(self.db_select_btn)
        
        # CSV导入按钮
        self.import_csv_btn = PushButton(self)
        self.import_csv_btn.setText("导入CSV")
        self.horizontalLayout2.addWidget(self.import_csv_btn)
        
        # SQL执行按钮
        self.execute_btn = PushButton(self)
        self.execute_btn.setText("执行SQL")
        self.horizontalLayout2.addWidget(self.execute_btn)
        
        # SQL输入框
        self.query_edit = QTextEdit(self)
        self.query_edit.setPlaceholderText("输入SQL查询语句...")
        self.query_edit.setMaximumHeight(100)
        self.verticalLayout.addWidget(self.query_edit)
        
        # 连接通用事件
        self.setup_connections()

    def setup_connections(self):
        """设置基本的事件连接"""
        self.execute_btn.clicked.connect(self.execute_query)
        self.import_csv_btn.clicked.connect(self.import_csv)
    
    def execute_query(self):
        """执行SQL查询"""
        query = self.query_edit.toPlainText()
        if not query.strip():
            self.show_message("错误", "请输入有效的SQL查询语句")
            return
            
        try:
            cursor = execute_query(query)
            results = cursor.fetchall()
            if results and cursor.description:
                columns = [desc[0] for desc in cursor.description]
                self.df = pd.DataFrame(results, columns=columns)
                self.display_dataframe(self.df)
                self.textedit.setText(f"查询成功！返回{len(results)}条记录 - 列名：{', '.join(columns)}")
            else:
                self.textedit.setText("查询成功，但未返回任何结果")
        except Exception as e:
            self.show_message("查询错误", str(e))
            self.textedit.setText(f"查询错误: {str(e)}")
    
    def import_csv(self):
        """导入CSV文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择CSV文件", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.display_dataframe(self.df)
                self.textedit.setText(f"成功导入CSV文件: {os.path.basename(file_path)}\n"
                                     f"形状: {self.df.shape[0]}行 x {self.df.shape[1]}列")
            except Exception as e:
                self.show_message("导入错误", str(e))
    
    def display_dataframe(self, df):
        """在表格中显示数据框"""
        if df is not None and not df.empty:
            model = PandasModel(df)
            self.tableView.setModel(model)
            self.tableView.resizeColumnsToContents()
    
    def create_canvas(self, width=8, height=6):
        """创建画布并添加到图形视图"""
        self.canvas = MplCanvas(self, width=width, height=height)
        self.graphic_scene = QGraphicsScene()
        self.graphic_scene.addWidget(self.canvas)
        self.graphicsView.setScene(self.graphic_scene)
        self.graphicsView.show()
        return self.canvas
    
    def show_message(self,title,content):
        InfoBar.warning(
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=True,
            position=InfoBarPosition.BOTTOM,
            duration=2000,    # won't disappear automatically
            parent=self
        )



class DataTablePage(BasePage):
    """数据表格页面 - 专注于数据导入和表格展示"""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        
        # 隐藏图形视图，扩展表格视图
        self.graphicsView.hide()
        
        # 添加数据描述按钮
        self.describe_btn = PushButton(self)
        self.describe_btn.setText("数据描述")
        self.horizontalLayout2.addWidget(self.describe_btn)
        
        # 添加数据过滤按钮
        self.filter_btn = PushButton(self)
        self.filter_btn.setText("数据过滤")
        self.horizontalLayout2.addWidget(self.filter_btn)
        
        # 连接信号
        self.describe_btn.clicked.connect(self.describe_data)
        self.filter_btn.clicked.connect(self.filter_data)
    
    def describe_data(self):
        """生成数据描述统计"""
        if self.df is None or self.df.empty:
            self.show_message("错误", "请先加载数据")
            return
            
        try:
            # 获取统计描述
            desc = self.df.describe(include='all').T
            desc['column'] = desc.index
            desc = desc[['column', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
            
            # 显示统计信息
            self.display_dataframe(desc)
            self.textedit.setText("数据描述统计：\n" + 
                                 f"行数: {self.df.shape[0]}\n" +
                                 f"列数: {self.df.shape[1]}\n" +
                                 f"数据类型:\n{self.df.dtypes.to_string()}")
        except Exception as e:
            self.show_message("处理错误", str(e))
    
    def filter_data(self):
        """添加数据过滤功能"""
        if self.df is None or self.df.empty:
            self.show_message("错误", "请先加载数据")
            return
        
        # 这里可以实现更复杂的过滤逻辑
        self.textedit.setText("数据过滤功能即将实现...\n" +
                             "您可以使用SQL查询来实现过滤:\n" +
                             "例如: SELECT * FROM your_table WHERE column > value")


class ChartPage(BasePage):
    """图表可视化页面"""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        
        # 隐藏文本区域，扩展图形区域
        self.tableView.hide()
        
        # 添加图表类型选择
        self.chart_label = QLabel("图表类型:", self)
        self.horizontalLayout2.addWidget(self.chart_label)
        
        self.chart_type = ComboBox(self)
        self.chart_type.addItems(["折线图", "柱状图", "散点图", "饼图", "热力图", "箱线图", "小提琴图"])
        self.horizontalLayout2.addWidget(self.chart_type)
        
        # 添加绘图按钮
        self.plot_btn = PushButton(self)
        self.plot_btn.setText("绘制图表")
        self.horizontalLayout2.addWidget(self.plot_btn)
        
        # 添加保存按钮
        self.save_btn = PushButton(self)
        self.save_btn.setText("保存图表")
        self.horizontalLayout2.addWidget(self.save_btn)
        
        # 连接信号
        self.plot_btn.clicked.connect(self.plot_chart)
        self.save_btn.clicked.connect(self.save_chart)
    
    def plot_chart(self):
        """根据选择的图表类型绘制图表"""
        if self.df is None or self.df.empty:
            self.show_message("错误", "请先加载数据")
            return
            
        try:
            chart_type = self.chart_type.currentText()
            
            # 创建画布
            canvas = self.create_canvas()
            
            # 设置美观的样式
            sns.set_theme(style="whitegrid")
            
            # 根据图表类型绘制
            if chart_type == "折线图":
                # 对每个数值列绘制折线图
                numeric_cols = self.df.select_dtypes(include=np.number).columns[:5]  # 限制为前5列
                if len(numeric_cols) == 0:
                    self.show_message("错误", "没有数值列可以绘制折线图")
                    return
                    
                for col in numeric_cols:
                    sns.lineplot(x=self.df.index, y=self.df[col], ax=canvas.axes, label=col)
                
                canvas.axes.set_title("折线图")
                canvas.axes.set_xlabel("索引")
                canvas.axes.set_ylabel("值")
                canvas.axes.legend()
                
            elif chart_type == "柱状图":
                # 获取一些数值列
                numeric_cols = self.df.select_dtypes(include=np.number).columns[:5]  # 限制为前5列
                if len(numeric_cols) == 0:
                    self.show_message("错误", "没有数值列可以绘制柱状图")
                    return
                
                # 使用第一列数据绘制柱状图
                sns.barplot(x=self.df.index[:20], y=self.df[numeric_cols[0]][:20], ax=canvas.axes)
                canvas.axes.set_title(f"{numeric_cols[0]} 柱状图")
                canvas.axes.set_xlabel("索引")
                canvas.axes.set_ylabel(numeric_cols[0])
                
            elif chart_type == "散点图":
                # 获取前两个数值列绘制散点图
                numeric_cols = self.df.select_dtypes(include=np.number).columns
                if len(numeric_cols) < 2:
                    self.show_message("错误", "至少需要两个数值列才能绘制散点图")
                    return
                
                sns.scatterplot(x=self.df[numeric_cols[0]], y=self.df[numeric_cols[1]], ax=canvas.axes)
                canvas.axes.set_title("散点图")
                canvas.axes.set_xlabel(numeric_cols[0])
                canvas.axes.set_ylabel(numeric_cols[1])
                
            elif chart_type == "饼图":
                # 选择一个分类列做饼图
                categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) == 0:
                    # 如果没有分类列，使用第一个数值列
                    numeric_cols = self.df.select_dtypes(include=np.number).columns
                    if len(numeric_cols) == 0:
                        self.show_message("错误", "没有合适的列可以绘制饼图")
                        return
                    
                    # 对数值列进行分组统计
                    value_counts = self.df[numeric_cols[0]].value_counts()[:10]  # 限制为前10个值
                else:
                    # 使用第一个分类列
                    value_counts = self.df[categorical_cols[0]].value_counts()[:10]  # 限制为前10个类别

                value_counts.plot.pie(autopct='%1.1f%%', ax=canvas.axes)
                canvas.axes.set_title("饼图")
                canvas.axes.set_ylabel("")
                
            elif chart_type == "热力图":
                # 计算相关性矩阵
                numeric_df = self.df.select_dtypes(include=np.number)
                if numeric_df.shape[1] < 2:
                    self.show_message("错误", "至少需要两个数值列才能绘制热力图")
                    return
                
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=canvas.axes)
                canvas.axes.set_title("相关性热力图")
                
            elif chart_type == "箱线图":
                # 获取数值列
                numeric_cols = self.df.select_dtypes(include=np.number).columns[:5]  # 限制为前5列
                if len(numeric_cols) == 0:
                    self.show_message("错误", "没有数值列可以绘制箱线图")
                    return
                
                # 绘制箱线图
                sns.boxplot(data=self.df[numeric_cols], ax=canvas.axes)
                canvas.axes.set_title("箱线图")
                canvas.axes.set_xlabel("特征")
                canvas.axes.set_ylabel("值")
                
            elif chart_type == "小提琴图":
                # 获取数值列
                numeric_cols = self.df.select_dtypes(include=np.number).columns[:3]  # 限制为前3列
                if len(numeric_cols) == 0:
                    self.show_message("错误", "没有数值列可以绘制小提琴图")
                    return
                
                # 绘制小提琴图
                sns.violinplot(data=self.df[numeric_cols], ax=canvas.axes)
                canvas.axes.set_title("小提琴图")
                canvas.axes.set_xlabel("特征")
                canvas.axes.set_ylabel("值")
            
            # 调整布局并绘制
            canvas.figure.tight_layout()
            canvas.draw()
            
            # 显示解释
            self.textedit.setText(f"已生成{chart_type}\n"
                               f"数据形状: {self.df.shape[0]}行 x {self.df.shape[1]}列\n"
                               f"数据列: {', '.join(self.df.columns.tolist()[:5])}等")
                               
        except Exception as e:
            self.show_message("图表错误", str(e))
            self.textedit.setText(f"绘图错误: {str(e)}")
    
    def save_chart(self):
        """保存当前图表"""
        if self.canvas is None:
            self.show_message("错误", "没有图表可以保存")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图表", "", "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)")
        if file_path:
            try:
                self.canvas.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                self.show_message("成功", f"图表已保存到 {file_path}")
            except Exception as e:
                self.show_message("保存错误", str(e))


class StatisticsPage(BasePage):
    """统计分析页面"""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        
        # 添加统计分析功能按钮
        self.summary_btn = PushButton(self)
        self.summary_btn.setText("数据摘要")
        self.horizontalLayout2.addWidget(self.summary_btn)
        
        self.correlation_btn = PushButton(self)
        self.correlation_btn.setText("相关性分析")
        self.horizontalLayout2.addWidget(self.correlation_btn)
        
        self.group_btn = PushButton(self)
        self.group_btn.setText("分组统计")
        self.horizontalLayout2.addWidget(self.group_btn)
        
        # 连接信号
        self.summary_btn.clicked.connect(self.show_summary)
        self.correlation_btn.clicked.connect(self.show_correlation)
        self.group_btn.clicked.connect(self.show_group_stats)
    
    def show_summary(self):
        """显示数据摘要"""
        if self.df is None or self.df.empty:
            self.show_message("错误", "请先加载数据")
            return
            
        try:
            # 创建一个摘要文本
            summary = (f"数据摘要:\n\n"
                     f"形状: {self.df.shape[0]}行 x {self.df.shape[1]}列\n\n"
                     f"列名: {', '.join(self.df.columns.tolist())}\n\n"
                     f"数据类型:\n{self.df.dtypes.to_string()}\n\n"
                     f"缺失值统计:\n{self.df.isnull().sum().to_string()}\n\n"
                     f"数值型列统计:\n{self.df.describe().to_string()}\n\n")
            
            # 显示在文本区域
            self.textedit.setText(summary)
            
            # 创建画布显示数据类型分布
            canvas = self.create_canvas()
            
            # 统计各类型列数
            dtype_counts = self.df.dtypes.map(lambda x: x.name).value_counts()
            
            # 绘制类型分布饼图
            dtype_counts.plot.pie(autopct='%1.1f%%', ax=canvas.axes)
            canvas.axes.set_title("数据类型分布")
            canvas.axes.set_ylabel("")
            
            # 调整布局并绘制
            canvas.figure.tight_layout()
            canvas.draw()
            
            # 显示描述统计表格
            self.display_dataframe(self.df.describe().T)
            
        except Exception as e:
            self.show_message("处理错误", str(e))
    
    def show_correlation(self):
        """显示相关性分析"""
        if self.df is None or self.df.empty:
            self.show_message("错误", "请先加载数据")
            return
            
        try:
            # 获取数值列进行相关性分析
            numeric_df = self.df.select_dtypes(include=np.number)
            
            if numeric_df.shape[1] < 2:
                self.show_message("错误", "至少需要两个数值列才能进行相关性分析")
                return
            
            # 计算相关性矩阵
            corr = numeric_df.corr()
            
            # 显示相关性表格
            self.display_dataframe(corr)
            
            # 创建画布显示热力图
            canvas = self.create_canvas()
            
            # 绘制热力图
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=canvas.axes)
            canvas.axes.set_title("相关性热力图")
            
            # 调整布局并绘制
            canvas.figure.tight_layout()
            canvas.draw()
            
            # 找出高相关性变量
            high_corr = (corr.abs() > 0.7) & (corr.abs() < 1.0)
            high_corr_pairs = []
            
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if high_corr.iloc[i, j]:
                        high_corr_pairs.append(f"{corr.columns[i]} & {corr.columns[j]}: {corr.iloc[i, j]:.2f}")
            
            if high_corr_pairs:
                self.textedit.setText("相关性分析:\n\n高相关性变量对:\n" + "\n".join(high_corr_pairs))
            else:
                self.textedit.setText("相关性分析:\n\n未发现高相关性变量对 (|r| > 0.7)")
                
        except Exception as e:
            self.show_message("处理错误", str(e))
    
    def show_group_stats(self):
        """显示分组统计"""
        if self.df is None or self.df.empty:
            self.show_message("错误", "请先加载数据")
            return
            
        try:
            # 检查是否有分类列
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            
            if len(categorical_cols) == 0 or len(numeric_cols) == 0:
                self.show_message("错误", "分组统计需要至少一个分类列和一个数值列")
                return
            
            # 使用第一个分类列和第一个数值列进行分组统计
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # 分组计算均值、计数和标准差
            group_stats = self.df.groupby(cat_col)[num_col].agg(['mean', 'count', 'std']).reset_index()
            
            # 显示分组统计表格
            self.display_dataframe(group_stats)
            
            # 创建画布
            canvas = self.create_canvas()
            
            # 绘制分组柱状图
            sns.barplot(x=cat_col, y=num_col, data=self.df, ax=canvas.axes)
            canvas.axes.set_title(f"{cat_col} 分组 {num_col} 均值")
            canvas.axes.set_xlabel(cat_col)
            canvas.axes.set_ylabel(f"{num_col} 均值")
            
            # 调整布局并绘制
            canvas.figure.tight_layout()
            canvas.draw()
            
            # 显示文本说明
            self.textedit.setText(f"分组统计: {cat_col} → {num_col}\n\n"
                               f"分组总数: {group_stats.shape[0]}\n\n"
                               f"最大均值组: {group_stats.loc[group_stats['mean'].idxmax()][cat_col]} "
                               f"({group_stats['mean'].max():.2f})\n\n"
                               f"最小均值组: {group_stats.loc[group_stats['mean'].idxmin()][cat_col]} "
                               f"({group_stats['mean'].min():.2f})")
                
        except Exception as e:
            self.show_message("处理错误", str(e))


class PandasModel(QAbstractTableModel):
    """用于在表格视图中显示Pandas DataFrame的模型"""
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self._data.columns[col])
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return str(self._data.index[col])
        return None


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib画布包装器，用于在Qt中显示matplotlib图形"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        plt.style.use('ggplot')  # 使用美观的样式
        
        # 设置字体
        plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)  # 添加子图
        super(MplCanvas, self).__init__(fig)
>>>>>>> 5e7640a (version-1.1)
