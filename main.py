# coding:utf-8
import sys
from PySide6.QtWidgets import QApplication
from setupmainwindow import Window



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec()
