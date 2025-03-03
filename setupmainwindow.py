from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QIcon, QDesktopServices
from PySide6.QtWidgets import QApplication, QFrame, QStackedWidget, QHBoxLayout, QWidget

# 第三方库
from qfluentwidgets import (
    NavigationInterface, NavigationItemPosition, NavigationWidget, MessageBox,
    isDarkTheme, setTheme, Theme, setThemeColor, qrouter, NavigationAvatarWidget, FluentIcon as FIF
)
from qframelesswindow import FramelessWindow, StandardTitleBar

# 自定义模块
from page.pagewidget import Widget
from page.home import HomePage
from page.machinelearning import MachineLearningPage
from page.SVM import SVMPage
from page.datacollection import DataCollectionPage
from page.randomforest import RandomForestPage
from page.dataanalysis import DataAnalysisPage
from page.datavisualizer import DataVisualizer
from page.ELM import ELMPage
from page.RBF import RBFPage
from page.deeplearning import DeepLearningPage
from page.crackidentification import CrackIdentificationPage
from page.common.signal_bus import signalBus




class Window(FramelessWindow):

    def __init__(self):
        super().__init__()
        self.setTitleBar(StandardTitleBar(self))
        
        # 初始化基础组件
        self.hBoxLayout = QHBoxLayout(self)
        self.navigationInterface = NavigationInterface(self, showMenuButton=True, collapsible=False)
        self.stackWidget = QStackedWidget(self)
        
        # 连接卡片点击信号
        signalBus.switchToSampleCard.connect(self.handleCardNavigation)

        # create sub interface
        self.homeInterface = HomePage(self)
        self.datavisualizerInterface = DataVisualizer('DataVisualizer-Interface', self)
        self.dataanalysisInterface = DataAnalysisPage('DataAnalysis-Interface', self)
        self.datacollectionInterface = DataCollectionPage('DataCollection Interface', self)
        self.crackidentificationInterface = CrackIdentificationPage('CrackIdentification Interface', self)
        self.folderInterface = Widget('Folder Interface', self)
        self.settingInterface = Widget('Setting Interface', self)
        self.machinelearningInterface = MachineLearningPage('Machinelearning-Interface', self)

        # 机器学习子页面配置
        self.ml_subpages = [
            (SVMPage, '支持向量机'),
            (RandomForestPage, '随机森林'), 
            (ELMPage, '极限学习机'),
            (RBFPage, '径向基函数'),
            (Widget, '待实现')  # 或其他合适的页面类
        ]

        self.deeplearningInterface = DeepLearningPage('Deeplearning-Interface', self)

        # initialize layout
        self.initLayout()

        # add items to navigation interface

        self.initNavigation()

        self.initWindow()

    def initLayout(self):
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(0, self.titleBar.height(), 0, 0)
        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addWidget(self.stackWidget)
        self.hBoxLayout.setStretchFactor(self.stackWidget, 1)

    def initNavigation(self):
        # enable acrylic effect
        # self.navigationInterface.setAcrylicEnabled(True)
        self.addSubInterface(self.homeInterface, FIF.HOME, '主页')
        self.addSubInterface(self.datavisualizerInterface, FIF.VIEW, '数据可视化')
        
        self.addSubInterface(self.dataanalysisInterface, FIF.APPLICATION, '数据分析')
        self.addSubInterface(self.datacollectionInterface, FIF.MUSIC, '数据处理')

        self.addSubInterface(self.crackidentificationInterface, FIF.VIDEO, '裂隙识别')

        self.addSubInterface(self.machinelearningInterface, FIF.ROBOT, '机器学习', NavigationItemPosition.SCROLL)

        # 动态创建机器学习子页面
        for idx, (PageClass, title) in enumerate(self.ml_subpages, 1):
            interface = PageClass(f'Machinelearning Interface 1-{idx}', self)
            self.addSubInterface(
                interface,
                FIF.ROBOT,
                title,
                parent=self.machinelearningInterface
            )
        self.addSubInterface(self.deeplearningInterface, FIF.ROBOT, '深度学习', NavigationItemPosition.SCROLL)
        # 分界线
        self.navigationInterface.addSeparator()

        # add navigation items to scroll area
        self.addSubInterface(self.folderInterface, FIF.FOLDER, '文件', NavigationItemPosition.SCROLL)
        # for i in range(1, 21):
        #     self.navigationInterface.addItem(
        #         f'folder{i}',
        #         FIF.FOLDER,
        #         f'Folder {i}',
        #         lambda: print('Folder clicked'),
        #         position=NavigationItemPosition.SCROLL
        #     )

        # add custom widget to bottom
        self.navigationInterface.addWidget(
            routeKey='avatar',
            widget=NavigationAvatarWidget('用户', 'resource/shoko.png'),
            onClick=self.showMessageBox,
            position=NavigationItemPosition.BOTTOM,
        )

        self.addSubInterface(self.settingInterface, FIF.SETTING, 'Settings', NavigationItemPosition.BOTTOM)

        # !IMPORTANT: don't forget to set the default route key if you enable the return button
        qrouter.setDefaultRouteKey(self.stackWidget, self.homeInterface.objectName())

        # set the maximum width
        # self.navigationInterface.setExpandWidth(300)

        self.stackWidget.currentChanged.connect(self.onCurrentInterfaceChanged)
        self.stackWidget.setCurrentIndex(0)

        # always expand
        # self.navigationInterface.setCollapsible(False)

    def initWindow(self):
        # 窗口基础配置
        self.setWindowTitle('SciVisualizer')
        self.setWindowIcon(QIcon('resource/lxd.jpg'))
        self.titleBar.setAttribute(Qt.WA_StyledBackground)

        # 自适应屏幕尺寸
        screen = QApplication.primaryScreen().availableGeometry()
        window_width = min(1920, screen.width() - 100)
        window_height = min(1080, screen.height() - 100)
        self.resize(window_width, window_height)
        
        # 居中显示
        x = (screen.width() - window_width) // 2
        y = (screen.height() - window_height) // 2
        self.move(x, y)

    def addSubInterface(self, interface, icon, text: str, position=NavigationItemPosition.TOP, parent=None):
        """添加子界面
        Args:
            interface: 要添加的界面实例
            icon: 导航图标 
            text: 导航显示文本
            position: 导航项位置，默认为顶部
            parent: 父级导航项
        """
        self.stackWidget.addWidget(interface)
        self.navigationInterface.addItem(
            routeKey=interface.objectName(),
            icon=icon,
            text=text,
            onClick=lambda: self.switchTo(interface),
            position=position,
            tooltip=text,
            parentRouteKey=parent.objectName() if parent else None
        )

    def switchTo(self, widget):
        self.stackWidget.setCurrentWidget(widget)

    def handleCardNavigation(self, routeKey, _):
        """处理卡片导航请求"""
        target = self.findChild(QWidget, routeKey)
        if target:
            self.switchTo(target)

    def onCurrentInterfaceChanged(self, index):
        widget = self.stackWidget.widget(index)
        self.navigationInterface.setCurrentItem(widget.objectName())

        # !IMPORTANT: This line of code needs to be uncommented if the return button is enabled
        # qrouter.push(self.stackWidget, widget.objectName())

    def showMessageBox(self):
        w = MessageBox(
            '支持作者🥰',
            '个人开发不易，如果这个项目帮助到了您，可以考虑请作者喝一瓶快乐水🥤。您的支持就是作者开发和维护项目的动力🚀',
            self
        )
        w.yesButton.setText('来啦老弟')
        w.cancelButton.setText('下次一定')

        if w.exec():
            QDesktopServices.openUrl(QUrl("https://afdian.net/a/zhiyiYo"))
