# coding:utf-8
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from setupmainwindow import Window

def initialize_app():
    app = QApplication(sys.argv)
    
    # 设置全局字体大小
    font = QFont()
    font.setPointSize(12)
    app.setFont(font)
    
    # 加载样式表
    load_stylesheet(app)
    
    return app

def load_stylesheet(app):
    style_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource", "style.qss")
    try:
        if os.path.exists(style_file):
            with open(style_file, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
        else:
            print(f"警告：样式表文件不存在: {style_file}")
    except Exception as e:
        print(f"加载样式表时出错: {e}")

if __name__ == '__main__':
    app = initialize_app()
    w = Window()
    w.show()
    sys.exit(app.exec())
