# coding: utf-8
from PySide6.QtCore import QObject, Signal


class SignalBus(QObject):
    """ Signal bus """

    switchToSampleCard = Signal(str, int)
    # 移除了未使用的信号
    # micaEnableChanged = Signal(bool)
    # supportSignal = Signal()


signalBus = SignalBus()